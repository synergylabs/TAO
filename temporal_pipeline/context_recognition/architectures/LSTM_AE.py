'''
This file contains architecture for LSTM AE
'''

import os
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import pandas as pd

MAX_SAMPLE_COUNT = 5000


def init_hidden(x, hidden_size, device, num_dir: int = 1):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        device: 'cuda:x' or 'cpu'
        num_dir: (int): number of directions in LSTM
        xavier: (bool): weather or not use xavier initialization
    """

    return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


class LSTMEncoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size, device):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features  # The number of expected features(= dimension size) in the input x
        self.hidden_size = no_features*2
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        # self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.device = device
        self.LSTM1 = nn.LSTM(
            input_size=self.no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            # bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size,self.embedding_size),
            nn.Sigmoid())

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x = x.view(-1, self.seq_len, self.no_features)
        h_0, c_0 = (init_hidden(x, self.hidden_size, self.device,1),
                    init_hidden(x, self.hidden_size, self.device,1))
        x, (hidden_state, cell_state) = self.LSTM1(x, (h_0, c_0))
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]
        out = self.fc(last_lstm_layer_hidden_state)
        return out


# (2) Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, embedding_size, output_size, device):
        super().__init__()

        self.seq_len = seq_len
        # self.no_features = embedding_size
        # self.hidden_size = (2 * embedding_size)
        self.hidden_size = 2*embedding_size
        self.output_size = output_size
        self.device = device
        self.LSTM1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.Sigmoid()
        )
        # self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        z, (hidden_state, cell_state) = self.LSTM1(z)
        z = z.reshape((-1, self.seq_len , self.hidden_size))
        z = self.linear(z)
        out = z.view(-1, self.seq_len * self.output_size)
        # x = self.fc(x)
        # out = x.view(-1, self.seq_len * self.output_size)
        return out


# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE_Network(nn.Module):
    def __init__(self, input_size, embedding_size,
                 run_config, logger=None):
        super().__init__()

        data_sample = run_config.data_sample

        self.config = run_config
        self.logger = logger
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.no_features = len(run_config.activity_labels)
        self.seq_len = int(len(data_sample) / len(run_config.activity_labels))
        self.stacked_input = run_config.stacked_input

        self.encoder = LSTMEncoder(self.seq_len, self.no_features, self.embedding_size, run_config.device)
        self.decoder = LSTMDecoder(self.seq_len, self.embedding_size, self.no_features, run_config.device)

        # self.criterion = nn.MSELoss(reduction='sum')
        # self.epochs = run_config.num_epochs_ae
        # self.learning_rate = run_config.lr_ae
        # self.patience = run_config.lstm_patience
        # self.max_grad_norm = run_config.lstm_max_grad_norm
        # self.every_epoch_print = run_config.log_interval_ae

    # def loss_function(self, reconstructed_x, x):
    #     '''
    #     Loss function for this model
    #     :return: loss function which takes x_pred, and x_true as input
    #     '''
    #     return self.criterion(reconstructed_x, x)

    def encode(self, x):
        """
        Encode a batch of input data in torch format
        :param x: torch.tensor for input batch
        :return: embedded tensor
        """
        if x.shape[0] > MAX_SAMPLE_COUNT:
            # encode in chunks to avoid our of memory error
            x_e = torch.empty((x.shape[0], self.embedding_size))
            for i in range(0, (x.shape[0] // MAX_SAMPLE_COUNT) + 1):
                x_e_batch = self.encoder(x[i * (MAX_SAMPLE_COUNT): (i + 1) * MAX_SAMPLE_COUNT, :])
                x_e[i * (MAX_SAMPLE_COUNT): (i + 1) * MAX_SAMPLE_COUNT, :] = x_e_batch.squeeze().cpu().detach()
                del x_e_batch
            return x_e.to(self.config.device)
        else:
            return self.encoder(x)

    def decode(self, z):
        """
        Decode a batch of embeddings from torch format
        :param z: torch.tensor for embedding batch
        :return: recreated input
        """
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save(self):
        """
        Save created encoder and decode models
        :return: None
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.encoder, f'{model_dir}/LSTM_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
        torch.save(self.decoder, f'{model_dir}/LSTM_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')

    def load(self):
        """
        Load older encoder/decoder model for given experiment
        :return:
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        try:
            self.encoder = torch.load(f'{model_dir}/LSTM_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
            self.decoder = torch.load(f'{model_dir}/LSTM_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')
            self.logger.info("Loaded pretrained model from experiment cache...")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"LSTM AE Model for experiment {self.config.experiment} does not exists..")
            else:
                print(f"LSTM AE Model for experiment {self.config.experiment} does not exists..")
            return False
