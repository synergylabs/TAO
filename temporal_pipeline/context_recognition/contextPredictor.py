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

from context_recognition.dataloaders import load_data_parser
from context_recognition.architectures import fetch_re_model
from context_recognition.representationTrainer import representationTrainer
from context_recognition.clustering import fetch_cluster_model
from context_recognition.clusterTrainer import clusterTrainer
from context_recognition.labelling import fetch_labeler
from context_recognition.contextLabeler import contextLabeler


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
        self.request_buffer_ts = []
        self.request_buffer_act = []
        self.activity_label_count = len(self.config.activity_labels)

    def predict(self, X):
        cluster_labels = []
        if self.model_cluster.is_input_raw:
            cluster_ids = self.model_cluster.predict(X)
        else:
            Z = self.model_re.get_embedding(X)
            cluster_ids = self.model_cluster.predict(np.expand_dims(Z, 0))

        for id in cluster_ids:
            cluster_labels.append(self.context_labeler.get_cluster_label(id))

        if len(cluster_labels) > 1:
            return cluster_labels
        else:
            return cluster_labels[0]

    def process_request(self, request_json):
        for activity in request_json['activities']:
            self.request_buffer_ts.append(request_json['timestamp'])
            self.request_buffer_act.append(activity)

        self.min_ts = min(self.request_buffer_ts)
        self.max_ts = max(self.request_buffer_ts)
        if self.max_ts - self.min_ts > self.config.lag_parameter*60:
            # drop older values
            self.request_buffer_ts = self.request_buffer_ts[
                                     -60 * self.activity_label_count * self.config.lag_parameter:]
            self.request_buffer_act = self.request_buffer_act[
                                      -60 * self.activity_label_count * self.config.lag_parameter:]
            X_request = self.context_vector_from_buffer()
            response_string = self.predict(X_request)
        else:
            response_string = 'Unavailable'
            self.logger.info("Incomplete information to predict context")

        return response_string

    def empty_buffer(self):
        self.request_buffer_ts = []
        self.request_buffer_act = []

    def context_vector_from_buffer(self):
        '''
        Get context vector from request buffer based on run config
        :param buffer:
        :return:
        '''
        lag_parameter = self.config.lag_parameter
        merge_mins = self.config.merge_mins
        df_data = pd.DataFrame(zip(self.request_buffer_ts, self.request_buffer_act),
                               columns=['timestamp', 'activity_name'])

        df_data['Activity'] = True
        if not (self.config.merge_mins == -1):
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
            if self.config.merge_mins < 1:
                merge_seconds = int(self.config.merge_mins * 60)
                if merge_seconds < 1:
                    merge_seconds = 1
                df_data['timestamp'] = df_data['timestamp'].apply(
                    lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute, merge_seconds * (
                            x.second // merge_seconds)))
            else:
                merge_mins = int(self.config.merge_mins)
                df_data['timestamp'] = df_data['timestamp'].apply(
                    lambda x: pd.datetime(x.year, x.month, x.day, x.hour,
                                          merge_mins * (x.minute // merge_mins), 0))
            df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9

        df_data = pd.pivot_table(df_data, index=['timestamp'], columns='activity_name',
                                 values='Activity', aggfunc='count')
        df_data[df_data > 1.] = 1.
        df_data_columns = df_data.columns
        for label in self.config.activity_labels:
            if label not in df_data_columns:
                df_data[label] = 0.
        df_data = df_data[self.config.activity_labels]
        df_data = df_data.fillna(0.).reset_index()
        df_data = df_data.sort_values(by=['timestamp'])
        df_activities = df_data[self.config.activity_labels]
        df_activities['activity_vec'] = df_activities.apply(lambda row: np.array(row, dtype=float),
                                                            axis=1)
        df_activities.loc[:, 'timestamp'] = df_data['timestamp']
        df_activities = df_activities[['timestamp', 'activity_vec']]
        window_length = lag_parameter // merge_mins
        df_activities = df_activities.iloc[-window_length:, :]
        ctx_vector = np.concatenate(df_activities['activity_vec'].values)
        ctx_vector = ctx_vector.reshape(1, -1)
        return ctx_vector
