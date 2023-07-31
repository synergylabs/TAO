'''
This file contains all parsers for stack and merge technique in which we compress data granularity based on
compression parameter
'''
import torch.utils.data
import numpy as np
import os
import pickle
import pandas as pd
import glob
from utils import time_diff, get_activity_vector
from datetime import datetime, timedelta
from context_recognition.labelling.manualLabeler import manualontolist as ontolist

'''
This function parses extrasensory dataset and create a copy in cache
'''


def parse_extrasensory_dataset(rawdatasrc,
                               args=None,
                               cache_dir='cache/',
                               access_cache=False,
                               logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache

        cache_file = f'{cache_dir}/extrasensory_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'

        if access_cache:
            logger.info(f"Trying to fetch data from {cache_file}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Extrasensory Data from cache..")
                X, X_deduped, context_request_dict = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, context_request_dict
            else:
                logger.info(f"Cache file not available. Fetching from raw datafile")
        else:
            logger.info("Cache access not allowed. Fetching from raw datafile")

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            logger.info("Starting Data Fetch...")
            t_datafetch_start = datetime.now()
            df_data = pd.read_csv(rawdatasrc)
            df_data = df_data[['uuid', 'timestamp', 'isTrain'] + args.activity_labels]
            df_data = pd.melt(df_data, id_vars=['uuid', 'timestamp', 'isTrain'], var_name='activity_name',
                              value_name='Activity')
            df_data = df_data[df_data.Activity == True]
            if not (args.merge_mins == -1):
                df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
                if args.merge_mins < 1:
                    merge_seconds = int(args.merge_mins * 60)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute, merge_seconds * (
                                x.second // merge_seconds)))
                else:
                    merge_mins = int(args.merge_mins)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour,
                                              merge_mins * (x.minute // merge_mins), 0))
                df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            # convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(
                lambda x: ontolist.activity_mapping[args.dataset][x])

            df_data = pd.pivot_table(df_data, index=['uuid', 'timestamp', 'isTrain'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['uuid', 'timestamp'])

            X = None
            context_request_dict = dict()
            onto_activity_labels = []
            for activity_label in args.onto_activity_labels:
                if activity_label in df_data.columns:
                    onto_activity_labels.append(activity_label)
            context_request_dict['onto_activity_labels'] = onto_activity_labels
            # do uuid level separation and create dataset for training
            for uuid in df_data['uuid'].unique():
                df_uuid_data = df_data[df_data.uuid == uuid]
                df_uuid_activities = df_uuid_data[onto_activity_labels]
                df_uuid_activities['activity_vec'] = df_uuid_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
                df_uuid_activities.loc[:, 'isTrain'] = df_uuid_data['isTrain']
                df_uuid_activities = df_uuid_activities[['timestamp', 'activity_vec', 'isTrain']]

                x_uuid = []
                ctx_req_uuid = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_uuid_activities.shape[0], args.sliding_parameter):
                    df_window = df_uuid_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        is_train_window = False
                        if (df_window.isTrain.sum() > 1):
                            is_train_window = True
                            x_uuid.append(ctx_vector)
                        ctx_req_uuid.append(
                            [is_train_window, df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[uuid] = pd.DataFrame(ctx_req_uuid,
                                                          columns=['isTrain', 'start_timestamp', 'end_timestamp',
                                                                   'ctx_vector'])

                if X is None:
                    X = np.stack(x_uuid, axis=0)
                else:
                    X = np.concatenate([X, np.stack(x_uuid, axis=0)], axis=0)
                logger.info(f"Fetched data for id {uuid}.")

            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

            X_deduped = np.unique(X, axis=0)
            if access_cache:
                pickle.dump((X, X_deduped, context_request_dict), open(cache_file, 'wb'))
            else:
                ...

    return X, X_deduped, context_request_dict


'''
This function parses casas dataset and create a copy in cache
'''


def parse_casas_dataset(rawdatasrc,
                        args=None,
                        cache_dir='cache/',
                        access_cache=False,
                        logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache
        cache_file = f'{cache_dir}/casas_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'
        if access_cache:
            logger.info(f"Trying to fetch data from {cache_file}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Casas Data from cache..")
                X, X_deduped, context_request_dict = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, context_request_dict
            else:
                logger.info(f"Cache file not available. Fetching from raw datafile")
        else:
            logger.info("Cache access not allowed. Fetching from raw datafile")

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            logger.info("Starting Data Fetch...")
            t_datafetch_start = datetime.now()
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.Activity.apply(
                lambda x: x.strip().replace('r1.', '').replace('r2.', ''))
            if not (args.merge_mins == -1):
                df_data['timestamp'] = pd.to_datetime(df_data['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
                if args.merge_mins < 1:
                    merge_seconds = int(args.merge_mins * 60)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute, merge_seconds * (
                                x.second // merge_seconds)))
                else:
                    merge_mins = int(args.merge_mins)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour,
                                              merge_mins * (x.minute // merge_mins), 0))
                df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9

            # convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(
                lambda x: ontolist.activity_mapping[args.dataset][x])

            df_data = pd.pivot_table(df_data, index=['Home', 'timestamp', 'isTrain'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['Home', 'timestamp', 'isTrain'])

            X = None
            context_request_dict = dict()
            onto_activity_labels = []
            for activity_label in args.onto_activity_labels:
                if activity_label in df_data.columns:
                    onto_activity_labels.append(activity_label)
            context_request_dict['onto_activity_labels'] = onto_activity_labels

            for home in df_data['Home'].unique():
                df_home_data = df_data[df_data.Home == home]
                df_home_activities = df_home_data[onto_activity_labels]
                df_home_activities['activity_vec'] = df_home_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_home_activities.loc[:, 'timestamp'] = df_home_data['timestamp']
                df_home_activities.loc[:, 'isTrain'] = df_home_data['isTrain']
                df_home_activities = df_home_activities[['timestamp', 'activity_vec', 'isTrain']]

                x_home = []
                ctx_req_home = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_home_activities.shape[0], args.sliding_parameter):
                    df_window = df_home_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length):
                        # x_home.append(np.concatenate(df_window['activity_vec'].values))
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        is_train_window = False
                        if df_window.isTrain.sum() > 1:
                            is_train_window = True
                            x_home.append(ctx_vector)
                        ctx_req_home.append(
                            [is_train_window, df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[home] = pd.DataFrame(ctx_req_home,
                                                          columns=['isTrain', 'start_timestamp', 'end_timestamp',
                                                                   'ctx_vector'])
                if len(x_home) > 0:
                    if X is None:
                        X = np.stack(x_home, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_home, axis=0)], axis=0)

                logger.info(f"Fetched data for id {home}.")
            X[X > 0] = 1.
            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

        X_deduped = np.unique(X, axis=0)
        if access_cache:
            pickle.dump((X, X_deduped, context_request_dict), open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X, X_deduped, context_request_dict


'''
This function parses tsu dataset and create a copy in cache
'''


def parse_tsu_dataset(rawdatasrc,
                      args=None,
                      cache_dir='cache/',
                      access_cache=False,
                      logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache
        cache_file = f'{cache_dir}/tsu_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'
        if access_cache:
            logger.info(f"Trying to fetch data from {cache_file}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got TSU Data from cache..")
                X, X_deduped, X_labelled = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, X_labelled
            else:
                logger.info(f"Cache file not available. Fetching from raw datafile")
        else:
            logger.info("Cache access not allowed. Fetching from raw datafile")

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            logger.info("Starting Data Fetch...")
            t_datafetch_start = datetime.now()
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.Activity
            if not (args.merge_mins == -1):
                df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
                if args.merge_mins < 1:
                    merge_seconds = int(args.merge_mins * 60)
                    if merge_seconds < 1:
                        merge_seconds = 1
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute, merge_seconds * (
                                x.second // merge_seconds)))
                else:
                    merge_mins = int(args.merge_mins)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour,
                                              merge_mins * (
                                                      x.minute // merge_mins),
                                              0))
                df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9

            # convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(
                lambda x: ontolist.activity_mapping[args.dataset][x])

            df_data = pd.pivot_table(df_data, index=['session_id', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['session_id', 'timestamp'])

            X = None
            context_request_dict = dict()

            for session in df_data['session_id'].unique():
                df_session_data = df_data[df_data.session_id == session]
                df_session_activities = df_session_data[args.onto_activity_labels]
                df_session_activities['activity_vec'] = df_session_activities.apply(
                    lambda row: np.array(row, dtype=float),
                    axis=1)
                df_session_activities.loc[:, 'timestamp'] = df_session_data['timestamp']
                df_session_activities = df_session_activities[['timestamp', 'activity_vec']]

                x_session = []
                ctx_req_session = []
                window_length = int(args.lag_parameter / args.merge_mins)
                for row_idx in range(0, df_session_activities.shape[0], args.sliding_parameter):
                    df_window = df_session_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length):
                        # x_session.append(np.concatenate(df_window['activity_vec'].values))
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_session.append(ctx_vector)
                        ctx_req_session.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])

                context_request_dict[session] = pd.DataFrame(ctx_req_session,
                                                             columns=['start_timestamp', 'end_timestamp', 'ctx_vector'])
                if len(x_session) > 0:
                    if X is None:
                        X = np.stack(x_session, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_session, axis=0)], axis=0)
                logger.info(f"Fetched data for id {session}.")
            X[X > 0] = 1.
            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

        if access_cache:
            X_deduped = np.unique(X, axis=0)
            pickle.dump((X, X_deduped, context_request_dict), open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X, X_deduped, context_request_dict
