'''
This is main class for temporal clustering with combined training
'''
import torch.nn as nn
import torch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import traceback
from sklearn.model_selection import train_test_split

from context_recognition.architectures import TAE
from context_recognition.dataparsers import activityDataset


class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, input_size, embedding_size, run_config, logger):
        super().__init__()

        self.config = run_config
        self.logger = logger
        self.input_size = input_size
        self.embedding_size = embedding_size

        ## init with the pretrained autoencoder model
        self.tae = TAE(input_size, embedding_size, run_config)
        self.tae.load()
        self.centr_size = self.tae.n_hidden

        ## clustering model
        self.alpha_ = run_config.cnet_alpha
        self.n_clusters = run_config.cnet_n_clusters
        self.device = run_config.device
        self.similarity = run_config.cnet_similarity

        ## Initialize clusternet model
        # self.model = ClusterNet(self.args)
        self.optimizer_clu = torch.optim.SGD(
            self.parameters(), lr=self.config.lr_cluster, momentum=self.config.momentum
        )

    # def init_ae_model(self):
    #     tae_model = TAE()
    #     model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
    #     try:
    #         self.encoder = torch.load(f'{model_dir}/Temporal_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
    #         self.decoder = torch.load(f'{model_dir}/Temporal_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')
    #         return True
    #     except Exception as e:
    #         if self.logger:
    #             self.logger.error(f"Temporal AE Model for experiment {self.config.experiment} does not exists..")
    #         else:
    #             print(f"Temporal AE Model for experiment {self.config.experiment} does not exists..")
    #         exit(1)

    def load_dataset(self,X):
        X_train, X_test = train_test_split(X, test_size=self.config.train_test_ratio,shuffle=True)
        activity_dataset_train = activityDataset(X_train)
        activity_dataset_test = activityDataset(X_test)
        self.train_loader = torch.utils.data.DataLoader(activity_dataset_train,
                                                        batch_size=self.config.batch_size_cluster,
                                                        shuffle=self.config.shuffle)
        self.test_loader = torch.utils.data.DataLoader(activity_dataset_test,
                                                        batch_size=self.config.batch_size_cluster,
                                                        shuffle=self.config.shuffle)
        self.X_scaled = X_train
        self.args.serie_size = self.X_scaled.shape[1]

    def init_centroids(self, z):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        # z, _ = self.tae(x.squeeze().unsqueeze(1).detach())
        z_np = z.detach().cpu()
        assignments = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="precomputed"
        ).fit_predict(
            compute_similarity(z_np, z_np, similarity=self.similarity)
        )

        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )

        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignments) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

        self.centroids = nn.Parameter(centroids_)

    def forward(self, x):

        z, x_reconstr = self.tae(x)
        z_np = z.detach().cpu()

        similarity = compute_similarity(
            z, self.centroids, similarity=self.similarity
        )

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.nansum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.nansum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        return z, x_reconstr, Q, P

    def kl_loss_function(self, input, pred):
        out = input * torch.log((input) / (pred))
        return torch.mean(torch.sum(out, dim=1))

    def train_ClusterNET(self, epoch, verbose, cluster_loss_ratio=0.8):
        """
        Function for training one epoch of the DTC
        """
        self.train()
        train_loss_kl, train_loss_bce, train_loss = 0, 0, 0
        all_preds, all_gt = [], []
        for batch_idx, inputs in enumerate(self.train_loader):
            inputs = inputs.type(torch.FloatTensor).to(self.args.device)
            # all_gt.append(labels.cpu().detach())
            self.optimizer_clu.zero_grad()
            z, x_reconstr, Q, P = self.forward(inputs)
            loss_bce = self.loss_function(x_reconstr,inputs, reduction='mean')
            loss_KL = self.kl_loss_function(P, Q)

            total_loss = (1- cluster_loss_ratio)*loss_bce + (cluster_loss_ratio)*loss_KL
            total_loss.backward()
            self.optimizer_clu.step()

            preds = torch.max(Q, dim=1)[1]
            all_preds.append(preds.cpu().detach())
            train_loss += total_loss.item()
            train_loss_kl += loss_KL.item()
            train_loss_bce += loss_bce.item()
        if verbose:
            print(
                "For epoch ",
                epoch,
                " Loss is(BCE | KL | Weighted) : %.3f | %.3f | %.3f" %
                (train_loss_bce / (batch_idx + 1), train_loss_kl / (batch_idx + 1), train_loss / (batch_idx + 1)),
            )
        # all_gt = torch.cat(all_gt, dim=0).numpy()
        # all_preds = torch.cat(all_preds, dim=0).numpy()
        return train_loss / (batch_idx + 1)

    def fit_predict(self, X):

        # initialize centroids
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(self.args.device)
        self.init_centroids(X_tensor)

        ## train clustering model
        best_training_loss = np.inf
        print("Training full model ...")
        for epoch in range(self.args.max_epochs):
            self.load_dataset(X) # reload entire dataset
            epoch_train_loss = self.train_ClusterNET(
                epoch, verbose=self.config.verbose, cluster_loss_ratio=self.config.cluster_loss_ratio
            )
            if (epoch_train_loss) < best_training_loss:
                best_training_loss = epoch_train_loss
                print("Current Best Training Loss:", best_training_loss)

        ## get center prediction for Z values
        _,_,Q,_ = self.forward(X_tensor)
        Z_labels = np.argmax(Q, axis=1)
        return Z_labels

    def predict(self, Z):
        z_t = torch.from_numpy(Z).type(torch.FloatTensor).to(self.args.device)
        similarity = compute_similarity(
            z_t, self.centroids, similarity=self.similarity
        )
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.nansum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        return np.argmax(Q, axis=1)

    def get_context_representation(self):
        return self.centroids.detach().numpy()

    def save(self):
        """
        Save created encoder and decode models
        :return: None
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.tae, f'{model_dir}/ClusterNet_TAE_{self.input_size}_{self.embedding_size}.pt')
        torch.save(self, f'{model_dir}/ClusterNet_{self.input_size}_{self.embedding_size}.pt')

    def load(self):
        """
        Load older encoder/decoder model for given experiment
        :return:
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        try:
            self.tae = torch.load(f'{model_dir}/ClusterNet_TAE_{self.input_size}_{self.embedding_size}.pt')
            self.model = torch.load(f'{model_dir}/ClusterNet_{self.input_size}_{self.embedding_size}.pt')
            self.optimizer_clu = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr_cluster, momentum=self.config.momentum
            )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Temporal ClusterNet Model for experiment {self.config.experiment} does not exists..")
            else:
                print(f"Temporal ClusterNet Model for experiment {self.config.experiment} does not exists..")
            return False


def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))


def compute_similarity(z, centroids, similarity="EUC"):
    """
    Function that compute distance between a latent vector z and the clusters centroids.

    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]
    bs = z.shape[0]

    if similarity == "CID":
        CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
        CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
        z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
        CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
        CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
        return torch.transpose(mse, 0, 1) * CF

    elif similarity == "EUC":
        z = z.expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        return torch.transpose(mse, 0, 1)

    elif similarity == "COR":
        std_z = (
            torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_z = (
            torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        std_cen = (
            torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_cen = (
            torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        ## covariance
        z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
        cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
        prod_expec = torch.mean(
            z_expand * cen_expand, dim=2
        )  ## (bs , n_clusters)
        pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        return torch.sqrt(2 * (1 - pearson_corr))

