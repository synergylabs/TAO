'''
This is main prediction class to predict context based on pretrained representation and clustering models
'''

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial
import sys

from context_recognition.dataparsers import load_data_parser
from context_recognition.architectures import fetch_re_model
from context_recognition.representationTrainer import representationTrainer
from context_recognition.clustering import fetch_cluster_model
from context_recognition.clusterTrainer import clusterTrainer
from context_recognition.labelling import fetch_labeler
from context_recognition.contextLabeler import contextLabeler

# sys.path.append('../')
# from context_recognition.temporal_clustering.autoenc_cluster.architectures import FC_Encoder, FC_Decoder, CNN_1D_Encoder, CNN_1D_Decoder
from context_recognition.dataparsers import activityDataset


class contextPredictor:
    def __init__(self, run_config, logger):

        self.config = run_config
        self.input_size = run_config.input_size
        self.embedding_size = run_config.embedding_size
        self.logger = logger

        # get data parser

        data_parse_func = load_data_parser(run_config.dataset, run_config.parse_style, logger)
        self.data_parser = partial(data_parse_func, args=run_config,
                                   cache_dir=run_config.cache_dir,
                                   access_cache=run_config.access_cache,
                                   logger=logger)

        # fetch model architecture
        model_re_arch = fetch_re_model(model_name=run_config.model_re,
                                       logger=logger)

        # initialize model and load dataset
        self.model_re = representationTrainer(model_arch=model_re_arch,
                                              input_size=run_config.input_size,
                                              embedding_size=run_config.embedding_size,
                                              run_config=run_config,
                                              logger=logger)

        is_loaded = self.model_re.load()
        if not is_loaded:
            logger.error("Pretrained Model not available for representation trainer, exiting...")
            sys.exit(1)

        model_cluster_arch = fetch_cluster_model(run_config.model_cluster, logger)

        # initialize clustering model

        self.model_cluster = clusterTrainer(model_cluster=model_cluster_arch,
                                            input_size=run_config.input_size,
                                            embedding_size=run_config.embedding_size,
                                            run_config=run_config,
                                            logger=logger)
        is_loaded = self.model_cluster.load()
        if not is_loaded:
            logger.error("Pretrained Model not available for context clustering, exiting...")
            sys.exit(1)

        # fetch accurate labeling model
        labeler_model = fetch_labeler(labeler_name=run_config.model_labeler, logger=logger)

        # initialize context labeller

        self.context_labeler = contextLabeler(labeler_model=labeler_model,
                                              run_config=run_config,
                                              logger=logger)
        is_loaded = self.context_labeler.load()
        if not is_loaded:
            logger.error("Pretrained Model not available for context labeler, exiting...")
            sys.exit(1)

        # create an active array to support data caching and buffering
        self.request_buffer = np.array([None] * run_config.lag_parameter)
        self.buffer_idx = 0
        self.new_data_cache = []

        return None

    def predict(self, X):
        cluster_labels = []
        if self.model_cluster.is_input_raw:
            cluster_ids = self.model_cluster.predict(X)
        else:
            Z = self.model_re.get_embedding(X)
            cluster_ids = self.model_cluster.predict(np.expand_dims(Z,0))

        for id in cluster_ids:
            cluster_labels.append(self.context_labeler.get_cluster_label(id))

        if len(cluster_labels) > 1:
            return cluster_labels
        else:
            return cluster_labels[0]

    def process_request(self, request_json):

        self.request_buffer[self.buffer_idx] = request_json
        self.buffer_idx = (self.buffer_idx + 1) % self.config.lag_parameter

        response_string = 'Unavailable'
        if None in self.request_buffer:
            self.logger.info("Incomplete information to predict context")
        else:
            # order indexes such that history order is maintained
            idx_order = np.arange(self.buffer_idx,self.buffer_idx+self.config.lag_parameter,dtype=int) % self.config.lag_parameter
            X_request = self.data_parser(self.request_buffer[idx_order])
            if len(X_request.shape) ==1:
                X_request = np.expand_dims(X_request,0)
            response_string = self.predict(X_request)
            self.new_data_cache.append(X_request)

        return response_string











