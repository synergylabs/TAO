'''
This file contains architecture for CNN for representation learning
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import os


class CNN_1D_Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, l1_kernel_size, channel_mult=10):
        super(CNN_1D_Encoder, self).__init__()

        self.input_size = input_size
        self.l1_kernel_size=l1_kernel_size
        self.embedding_size = embedding_size
        self.channel_mult = channel_mult
        self.pooling = 4

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.channel_mult*1,
                kernel_size=l1_kernel_size,
                stride=l1_kernel_size,
            ),
            nn.MaxPool1d(self.pooling),
            nn.LeakyReLU(),
            # nn.Conv1d(
            #     in_channels=self.channel_mult*1,
            #     out_channels=self.channel_mult*2,
            #     kernel_size=5,
            #     stride=1,
            #     # padding=5,
            # ),
            # nn.LeakyReLU(),
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Sigmoid(),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1,1,self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, 1, self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_1D_Decoder(nn.Module):
    def __init__(self, embedding_size, output_size, l1_kernel_size, channel_mult=10):
        super(CNN_1D_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.output_size = output_size
        self.channel_mult = channel_mult
        self.l1_kernel_size=l1_kernel_size
        self.output_channels = 1
        self.fc_output_dim = 64
        #
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        # self.deconv = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose1d(
        #         in_channels= self.fc_output_dim,
        #         out_channels=channel_mult*2,
        #         kernel_size=6,
        #         stride=1,
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         in_channels=channel_mult * 2,
        #         out_channels=channel_mult * 1,
        #         kernel_size=5,
        #         stride=1,
        #     ),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose1d(
        #         in_channels=channel_mult * 1,
        #         out_channels=1,
        #         kernel_size=l1_kernel_size,
        #         stride=l1_kernel_size,
        #     ),
        #     nn.LeakyReLU()
        # )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.fc_output_dim,
                out_channels=channel_mult,
                kernel_size=l1_kernel_size,
                stride=1,
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=channel_mult,
                out_channels=1,
                kernel_size=l1_kernel_size,
                stride=1,
            ),
            nn.LeakyReLU()
        )
        self.flat_fts = self.get_flat_fts(self.deconv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid(),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1,self.fc_output_dim,1)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim,1)
        x = self.deconv(x)
        x = x.view(-1, self.flat_fts)
        return self.linear(x)



class CNN_AE_Network(nn.Module):
    def __init__(self, input_size, embedding_size,
                 run_config, logger=None):
        super(CNN_AE_Network, self).__init__()
        self.config=run_config
        self.logger = logger
        self.input_size=input_size
        self.embedding_size=embedding_size
        if self.config.mode=='train':
            self.encoder = CNN_1D_Encoder(input_size, embedding_size, run_config.cnn_kernel_size, run_config.cnn_channel_mult)
            self.decoder = CNN_1D_Decoder(embedding_size, input_size, run_config.cnn_kernel_size, run_config.cnn_channel_mult)
        else:
            is_model_loaded = self.load()
            self.is_model_loaded = True
        return None

    def encode(self, x):
        """
        Encode a batch of input data in torch format
        :param x: torch.tensor for input batch
        :return: embedded tensor
        """
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
        torch.save(self.encoder, f'{model_dir}/CNN_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
        torch.save(self.decoder, f'{model_dir}/CNN_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')

    def load(self):
        """
        Load older encoder/decoder model for given experiment
        :return:
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        try:
            self.encoder = torch.load(f'{model_dir}/CNN_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
            self.decoder = torch.load(f'{model_dir}/CNN_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')
            self.logger.info("Loaded pretrained model from experiment cache...")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"CNN AE Model for experiment {self.config.experiment} does not exists..")
            else:
                print(f"CNN AE Model for experiment {self.config.experiment} does not exists..")
            return False



