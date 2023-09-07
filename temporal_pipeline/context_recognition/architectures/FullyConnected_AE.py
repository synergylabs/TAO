'''
This file contains architecture for FCN for representation learning
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchinfo import summary
import numpy as np
import os

class FC_Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2

class FC_Decoder(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(embedding_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class FCN_AE_Network(nn.Module):

    def __init__(self, input_size, embedding_size,
                 run_config, logger=None):
        super(FCN_AE_Network, self).__init__()
        self.config = run_config
        self.logger = logger
        self.input_size = input_size
        self.embedding_size = embedding_size
        if self.config.mode=='train':
            self.encoder = FC_Encoder(input_size, embedding_size, run_config.fcn_hidden_size)
            self.decoder = FC_Decoder(embedding_size, input_size, run_config.fcn_hidden_size)
        else:
            self.is_model_loaded = self.load()
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
        torch.save(self.encoder, f'{model_dir}/FCN_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
        torch.save(self.decoder, f'{model_dir}/FCN_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')

    def load(self):
        """
        Load older encoder/decoder model for given experiment
        :return:
        """
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        try:
            self.encoder = torch.load(f'{model_dir}/FCN_AE_Encoder_{self.input_size}_{self.embedding_size}.pt')
            self.decoder = torch.load(f'{model_dir}/FCN_AE_Decoder_{self.input_size}_{self.embedding_size}.pt')
            self.logger.info("Loaded pretrained model from experiment cache...")
            return True
        except Exception as e:
            if hasattr(self,'logger'):
                self.logger.error(f"FCN AE Model for experiment {self.config.experiment} does not exists..")
            else:
                print(f"FCN AE Model for experiment {self.config.experiment} does not exists..")
            return False
