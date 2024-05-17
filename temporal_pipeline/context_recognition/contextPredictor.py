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
import os
import json
from itertools import combinations

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

        # initialize labeling information from clustered contexts based on ontological information
        self.cluster_labels = self.generate_cluster_labels()

        # create an active array to support data caching and buffering
        self.request_buffer_ts = []
        self.request_buffer_act = []
        self.activity_label_count = len(self.config.onto_activity_labels)

    def predict(self, X):
        cluster_labels = []
        if self.model_cluster.is_input_raw:
            cluster_ids = self.model_cluster.predict(X)
        else:
            Z = self.model_re.get_embedding(X)
            cluster_ids = self.model_cluster.predict(np.expand_dims(Z, 0))

        for id in cluster_ids:
            cluster_labels.append(self.cluster_labels[id])

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
        if self.max_ts - self.min_ts >= self.config.lag_parameter * 60:
            # drop older values
            eligible_idxs = np.where(np.array(self.request_buffer_ts)>self.max_ts-(60*self.config.lag_parameter))[0]
            self.request_buffer_ts = np.array(self.request_buffer_ts)[eligible_idxs].tolist()
            self.request_buffer_act = np.array(self.request_buffer_act)[eligible_idxs].tolist()
            X_request = self.context_vector_from_buffer()
            response_string = self.predict(X_request)
        else:
            response_string = 'Unavailable'
            self.logger.info("Incomplete information to predict context")

        activity_inputs = [
            f'{pd.to_datetime(self.request_buffer_ts[i], unit="s").strftime("%H:%M:%S")}:{self.request_buffer_act[i]}'
            for i in range(len(self.request_buffer_ts))]

        return response_string, activity_inputs

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
        for label in self.config.onto_activity_labels:
            if label not in df_data_columns:
                df_data[label] = 0.
        df_data = df_data[self.config.onto_activity_labels]
        df_data = df_data.fillna(0.).reset_index()
        df_data = df_data.sort_values(by=['timestamp'])
        df_activities = df_data[self.config.onto_activity_labels]
        df_activities['activity_vec'] = df_activities.apply(lambda row: np.array(row, dtype=float),
                                                            axis=1)
        df_activities.loc[:, 'timestamp'] = df_data['timestamp']
        df_activities = df_activities[['timestamp', 'activity_vec']]
        window_length = lag_parameter // merge_mins
        if df_activities.shape[0] >= window_length:
            df_activities = df_activities.iloc[-window_length:, :]
            ctx_vector = np.concatenate(df_activities['activity_vec'].values)
        else:
            leading_zeros_count  = (window_length-df_activities.shape[0])* len(self.config.onto_activity_labels)
            leading_zeros_arr = np.zeros(leading_zeros_count)
            ctx_vector = np.concatenate(df_activities['activity_vec'].values)
            ctx_vector = np.concatenate([leading_zeros_arr, ctx_vector])

        ctx_vector = ctx_vector.reshape(1, -1)
        return ctx_vector

    def generate_cluster_labels(self):
        '''
        Get cluster label from representation information
        Returns:
        '''
        repr_info_file = f'{self.config.cache_dir}/{self.config.experiment}/models/cluster_representation_info.json'
        if not os.path.exists(repr_info_file):
            self.logger.error(
                f'Clustering meta information not available at {repr_info_file}. Please copy this file from {repr_info_file.split("/models")[0]}/results/results_summary_xxx.json')

        df_onto = pd.read_csv(f'{self.config.ontology_labels_csv}', names=['activities', 'contexts'])
        df_onto['contexts'] = df_onto['contexts'].apply(lambda x: x.split(";"))
        onto_dict = df_onto.set_index('activities').to_dict()['contexts']

        dataset = self.config.dataset

        exp_results = json.load(open(repr_info_file, 'r'))
        direct_cluster_centers = exp_results['direct_labels']
        direct_cluster_labels = self.get_cluster_labels(direct_cluster_centers, dataset, onto_dict)
        decoded_cluster_centers = exp_results['decoded_labels']
        decoded_cluster_labels = self.get_cluster_labels(decoded_cluster_centers, dataset, onto_dict)
        cluster_labels = []
        for idx in range(len(decoded_cluster_labels)):
            if len(decoded_cluster_labels[idx]) > 0:
                cluster_labels.append(decoded_cluster_labels[idx])
            else:
                cluster_labels.append(direct_cluster_labels[idx])
        df_cluster_merge = pd.DataFrame(
            np.array([[f"{idx}:" + ','.join(cr) for idx, cr in enumerate(direct_cluster_labels)],
                      [f"{idx}:" + ','.join(cr) for idx, cr in enumerate(decoded_cluster_labels)],
                      [f"{idx}:" + ','.join(cr) for idx, cr in enumerate(cluster_labels)]]).T,
            columns=['direct', 'decoded', 'combination_1'])
        final_cluster_labels = df_cluster_merge['combination_1'].values
        final_cluster_labels = np.array([xr.split(":")[-1] for xr in final_cluster_labels])
        return final_cluster_labels

    def get_cluster_labels(self, cluster_centers, dataset, onto_dict):
        cluster_centers = [center.split(")__")[0].split("(")[-1] for center in cluster_centers]
        cluster_labels = [label_context_helper(center, activity_rename_mapping[dataset], onto_dict) for center in
                          cluster_centers]
        return cluster_labels


activity_rename_mapping = {
    'extrasensory': {
        'lying': 'LyingDown',
        'sitting': 'Sitting',
        'walking': 'Walking',
        'running': 'Running',
        'cycling': 'Cycling',
        'sleeping': 'Sleeping',
        'meeting': 'Meeting',
        'driving': 'Driving',
        'exercising': 'Hiking',
        'cooking': 'Cooking',
        'shopping': 'Shopping',
        'drinking': 'Drinking',
        'shower': 'Shower',
        'cleaning': 'VacuumHome',
        'laundry': 'VacuumHome',
        'clean_dishes': 'VacuumHome',
        'watching_tv': 'WatchingTv',
        'surfing_internet': 'ReadingOffice',
        'singing': 'Dancing',
        'talking': 'Talking',
        'office_work': 'TypingOffice',
        'eating': 'Eating',
        'toilet': 'Toilet',
        'grooming': 'Grooming',
        'dressing_up': 'Grooming',
        'stairs': 'ClimbingStairs',
        'standing': 'Standing',
        'meeting_coworkers': 'Meeting',
        'meeting_friends': 'Dancing',

    },
    'casas': {
        'step_out': 'StepOut',
        'none': 'None',
        'toilet': 'Toilet',
        'onphone': 'OnPhone',
        'grooming': 'Grooming',
        'step_in': 'StepIn',
        'lying': 'LyingDown',
        'drinking': 'Drinking',
        'watching_tv': 'WatchingTv',
        'dressing_up': 'Grooming',
        'taking_meds': 'Eating',
        'wakingup': 'Sleeping',
        'reading': 'TypingOffice',
        'cooking': 'Cooking',
        'eating': 'Eating',
        'shower': 'Shower',
        'sleeping': 'Sleeping',
        'office_work': 'SittingOffice',
        'dishes_home': 'VacuumHome',
        'meeting_friends': 'Dancing',
        'exercising': 'Running',
        'laundry_home': 'VacuumHome'
    },
}


def label_context_helper(cluster_representation, activity_renaming, onto_dict,
                         conf_activities=['sitting', 'standing', 'talking']):
    # print(cluster_representation)
    act_train = cluster_representation.split(">")
    act_train = [xr.split("+") for xr in act_train]

    unique_activities = set()
    for act_set in act_train:
        for activity in act_set:
            unique_activities.add(activity)
    for conf_act in conf_activities:
        try:
            unique_activities.remove(conf_act)
        except:
            ...

    if len(unique_activities) > 0:
        # print("More than conf activities in the set, removing conf activities for precise context labeling")
        act_train = [([activity for activity in act_set if (activity not in conf_activities)]) for act_set in act_train]
        # print(act_train)

    for i in range(len(act_train)):
        for j in range(len(act_train[i])):
            # print(act_train,act_train[i], act_train[i][j])
            if not act_train[i][j] == 'unknown':
                act_train[i][j] = activity_renaming[act_train[i][j]].lower()
            else:
                act_train[i][j] = 'none'
        act_train[i] = sorted(np.unique(act_train[i]).tolist())

    # for sequential contexts
    # try:
    seq_contexts = []
    for i in range(1, len(act_train)):
        set1, set2 = act_train[i - 1], act_train[i]
        for first_act in set1:
            for sec_act in set2:
                # print(f"seq: {first_act},{sec_act}")
                if not first_act == sec_act:
                    try:
                        seq_ctx = onto_dict[
                            f'{first_act}+{sec_act}']
                        if not (seq_ctx[0] == 'Unknown'):
                            seq_contexts += seq_ctx
                    except:
                        ...

    # for parallel contexts
    single_act_contexts = []
    par_contexts = []
    for act_set in act_train:
        if len(act_set) == 1:
            single_ctx = onto_dict[f"{act_set[0]}"]
            if not (single_ctx[0] == 'Unknown'):
                single_act_contexts += single_ctx
        else:
            for act1, act2 in combinations(act_set, 2):
                if not act1 == act2:
                    par_ctx = ['Unknown']
                    try:
                        par_ctx = onto_dict[f"{act1}_{act2}"]
                    except:
                        par_ctx = onto_dict[f"{act2}_{act1}"]
                    if not (par_ctx[0] == 'Unknown'):
                        par_contexts += par_ctx

    final_context_set = None
    if len(seq_contexts) > 0:
        final_context_set = np.unique(seq_contexts).tolist()
    elif len(par_contexts) > 0:
        final_context_set = np.unique(par_contexts).tolist()
    elif len(single_act_contexts) > 0:
        final_context_set = np.unique(single_act_contexts).tolist()
    else:
        all_contexts = []
        for act_set in act_train:
            for activity in act_set:
                single_ctx = onto_dict[f"{activity}"]
                if not (single_ctx[0] == 'Unknown'):
                    all_contexts += single_ctx
        final_context_set = np.unique(all_contexts).tolist()

    return final_context_set
