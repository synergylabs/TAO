'''
This is wrapper class to train representation learning models
'''
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../training/')
# from context_recognition.temporal_clustering.autoenc_cluster.architectures import FC_Encoder, FC_Decoder, CNN_1D_Encoder, CNN_1D_Decoder
from context_recognition.dataloaders import activityDataset


class representationTrainer:

    def __init__(self, model_arch, input_size, embedding_size, run_config, logger):
        self.device = run_config.device
        self.model = model_arch(input_size, embedding_size, run_config, logger)
        if run_config.use_pretrained_model:
            is_loaded = self.model.load()
            if not is_loaded:
                logger.error("Pretrained Model not available for representation trainer, using new model...")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=run_config.lr_ae)
        self.loss_function = self.bce_loss_function
        if hasattr(self.model, 'optimize'):
            self.optimizer = self.model.optimizer
        if hasattr(self.model, 'loss_function'):
            self.loss_function = self.model.loss_function
        self.config = run_config
        self.logger = logger

    def bce_loss_function(self, reconstructed_x, x):
        BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        return BCE

    def num_correct_predictions(self, reconstructed_x, x):
        recon_np = reconstructed_x.cpu().detach().numpy()
        recon_np[recon_np > 0.5] = 1.
        recon_np[recon_np <= 0.5] = 0.
        x_np = x.cpu().detach().numpy()
        num_correct_preds = 0.
        for i in range(x_np.shape[0]):
            num_correct_preds += np.all(x_np[i] == recon_np[i])
        return num_correct_preds, x_np.shape[0]

    def load_dataset(self, X):
        X_train, X_test = train_test_split(X, test_size=self.config.train_test_ratio_re, shuffle=True)
        activity_dataset_train = activityDataset(X_train)
        activity_dataset_test = activityDataset(X_test)
        self.train_loader = torch.utils.data.DataLoader(activity_dataset_train,
                                                        batch_size=self.config.batch_size_re,
                                                        shuffle=self.config.shuffle_re)
        self.test_loader = torch.utils.data.DataLoader(activity_dataset_test,
                                                       batch_size=self.config.batch_size_re,
                                                       shuffle=self.config.shuffle_re)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        train_correct_preds = 0.
        train_total_preds = 0.

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            # get absolute accuracy
            batch_correct_preds, batch_total_preds = self.num_correct_predictions(recon_batch, data)
            train_correct_preds += batch_correct_preds
            train_total_preds += batch_total_preds

            if batch_idx % self.config.log_interval_ae == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        epoch_loss = train_loss / len(self.train_loader.dataset)
        epoch_accuracy = train_correct_preds * 100 / train_total_preds
        self.logger.info('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, epoch_loss))
        self.logger.info('====> Epoch: {} Total Accuracy: {:.4f}'.format(
            epoch, epoch_accuracy))

        return epoch_loss, epoch_accuracy

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        test_correct_preds = 0.
        test_total_preds = 0.
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()
                batch_correct_preds, batch_total_preds = self.num_correct_predictions(recon_batch, data)
                test_correct_preds += batch_correct_preds
                test_total_preds += batch_total_preds

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = test_correct_preds * 100 / test_total_preds

        self.logger.info('====> Test set loss: {:.4f}'.format(test_loss))
        self.logger.info('====> Test set Accuracy: {:.4f}'.format(test_accuracy))
        return test_loss, test_accuracy

    def reconstruct(self, X):
        with torch.no_grad():
            X_t = activityDataset(X).x
            X_reconstructed_t = self.model(X_t)
            X_reconstructed = X_reconstructed_t.detach().numpy()
        return X_reconstructed

    def get_embedding(self, X):
        torch.cuda.empty_cache()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.config.device)
            Z_t = self.model.encode(X_t)
            Z_t = Z_t.squeeze()
            Z = Z_t.cpu().detach().numpy()
        return Z

    def get_reconstructed_input(self, Z):
        with torch.no_grad():
            Z_t = torch.from_numpy(Z.astype(np.float32)).to(self.config.device)
            if len(Z.shape) == 1:  # single sample only
                Z_t = Z_t.unsqueeze(0)
            X_t = self.model.decode(Z_t)
            X = X_t.cpu().detach().numpy()
        # get binary output, only accept high confidence values
        recon_threshold = self.config.reconstruction_conf_val
        X[X > recon_threshold] = 1
        X[X <= recon_threshold] = 0
        return X

    def save(self):
        self.model.save()

    def load(self):
        return self.model.load()
