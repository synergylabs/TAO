'''
This file contains architecture for TAE(Temporal AE) for representation learning
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import gc
import os

MAX_SAMPLE_COUNT = 5000


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, input_size, cnn_kernel_size, filter_1, filter_lstm, pooling):
        super().__init__()
        self.input_size = input_size
        self.cnn_kernel_size = cnn_kernel_size
        self.filter_1 = filter_1
        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.pooling = pooling
        self.n_hidden = None
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=self.cnn_kernel_size,
                stride=1,
                padding=5
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.pooling),
        )

        self.lstm_1 = nn.LSTM(
            input_size=self.filter_1,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm_2 = nn.LSTM(
            input_size=self.filter_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x = x.view(-1, 1, self.input_size)
        out_cnn = self.conv_layer(x)
        out_cnn = out_cnn.permute((0, 2, 1))
        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(
            out_lstm1.view(
                out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1
            ),
            dim=2,
        )
        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(
                features.shape[0], features.shape[1], 2, self.hidden_lstm_2
            ),
            dim=2,
        )

        if self.n_hidden == None:
            self.n_hidden = features.shape[1]
        return features


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, cnn_kernel_size, n_hidden=64, pooling=8):
        super().__init__()

        self.pooling = pooling
        self.n_hidden = n_hidden
        self.fc_output_dim = 64
        self.deconv_out_channels = self.n_hidden
        self.cnn_kernel_size = cnn_kernel_size

        # upsample
        self.up_layer = nn.Upsample(size=pooling)

        # get size of upsampled tensor

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.n_hidden * pooling,
                out_channels=self.deconv_out_channels,
                kernel_size=10,
                stride=1,
                padding=self.pooling // 2,
            ),
            nn.Sigmoid()
        )

    def forward(self, z):
        upsampled = self.up_layer(z)
        upsampled = upsampled.view(upsampled.shape[0], -1)
        upsampled = upsampled.unsqueeze(2)
        out_deconv = self.deconv_layer(upsampled)[
                     :, :, : self.pooling
                     ].contiguous()
        out_deconv = out_deconv.view(out_deconv.shape[0], -1)
        return out_deconv


class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, input_size, embedding_size, run_config, logger):
        super().__init__()

        self.logger = logger
        self.config = run_config
        self.input_size = input_size
        self.embedding_size = embedding_size
        if self.config.mode == 'train':
            self.pooling = run_config.tae_pool_layer_size
            self.filter_1 = run_config.tae_cnn_filter_count
            self.filter_lstm = run_config.tae_filter_lstm
            self.cnn_kernel_size = run_config.tae_cnn_kernel_size
            self.encoder = TAE_encoder(
                input_size=input_size,
                cnn_kernel_size=self.cnn_kernel_size,
                filter_1=self.filter_1,
                filter_lstm=self.filter_lstm,
                pooling=self.pooling,
            )
            n_hidden = self.get_hidden(input_size, run_config.device)
            self.embedding_size = n_hidden
            self.n_hidden = n_hidden
            self.decoder = TAE_decoder(
                cnn_kernel_size=self.cnn_kernel_size,
                n_hidden=self.n_hidden, pooling=self.pooling
            )
        else:
            self.is_model_loaded = self.load()
        return None

    def get_hidden(self, serie_size, device):
        a = torch.randn((1, 1, serie_size)).to(device)
        test_model = TAE_encoder(
            self.input_size,
            cnn_kernel_size=self.cnn_kernel_size,
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
            pooling=self.pooling,
        ).to(device)
        with torch.no_grad():
            _ = test_model(a)
        n_hid = test_model.n_hidden
        del test_model, a
        gc.collect()
        torch.cuda.empty_cache()
        return n_hid

    def encode(self, x):
        """
        Encode a batch of input data in torch format
        :param x: torch.tensor for input batch
        :return: embedded tensor
        """

        if x.shape[0] > MAX_SAMPLE_COUNT:
            # encode in chunks to avoid our of memory error
            x_e = torch.empty((x.shape[0], self.embedding_size))
            for i in range(0, (x.shape[0] // MAX_SAMPLE_COUNT)+1):
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
        if len(z.shape) == 2:
            z_t = z.unsqueeze(2)
        else:
            z_t = z
        return self.decoder(z_t)
        # if z_t.shape[0] > MAX_SAMPLE_COUNT:
        #     # encode in chunks to avoid our of memory error
        #     x_d = torch.empty((z_t.shape[0], self.input_size))
        #     for i in range(0, z_t.shape[0] + MAX_SAMPLE_COUNT, MAX_SAMPLE_COUNT):
        #         x_d_batch = self.decoder(z_t[i * MAX_SAMPLE_COUNT: (i + 1) * MAX_SAMPLE_COUNT, :, :])
        #         x_d[i * MAX_SAMPLE_COUNT: (i + 1) * MAX_SAMPLE_COUNT, :] = x_d_batch.squeeze().cpu().detach()
        #         del x_d_batch
        #     return x_d.to(self.config.device)
        # else:
        #     return self.decoder(z_t)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save(self):
        """
        Save created encoder and decode models
        :return: None
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}/models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.encoder, f'{model_dir}/Temporal_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
        torch.save(self.decoder, f'{model_dir}/Temporal_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')

    def load(self):
        """
        Load older encoder/decoder model for given experiment
        :return:
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}/models'
        try:
            self.encoder = torch.load(f'{model_dir}/Temporal_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
            self.decoder = torch.load(f'{model_dir}/Temporal_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Temporal AE Model for experiment {self.config.experiment} does not exists..")
            else:
                print(f"Temporal AE Model for experiment {self.config.experiment} does not exists..")
            return False

    # def get_embeddings(self,x):
    #     return self.encoder(x)
