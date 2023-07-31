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

test_ids = ['00EABED2-271D-49D8-B599-1D4A09240601',
            'csh102',
            '0A986513-7828-4D53-AA1F-E02D6DF9561B',
            'csh101',
            '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
            '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
            '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E',
            '098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
            '0BFC35E2-4817-4865-BFA7-764742302A2D']

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
            df_data = df_data[['uuid', 'timestamp'] + args.activity_labels]
            df_data = pd.melt(df_data, id_vars=['uuid', 'timestamp'], var_name='activity_name', value_name='Activity')
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

            df_data = pd.pivot_table(df_data, index=['uuid', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['uuid', 'timestamp'])

            X = None
            context_request_dict = dict()
            # do uuid level separation and create dataset for training
            for uuid in df_data['uuid'].unique():
                df_uuid_data = df_data[df_data.uuid == uuid]
                df_uuid_activities = df_uuid_data[args.onto_activity_labels]
                df_uuid_activities['activity_vec'] = df_uuid_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
                df_uuid_activities = df_uuid_activities[['timestamp', 'activity_vec']]

                x_uuid = []
                ctx_req_uuid = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_uuid_activities.shape[0], args.sliding_parameter):
                    df_window = df_uuid_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_uuid.append(ctx_vector)
                        ctx_req_uuid.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[uuid] = pd.DataFrame(ctx_req_uuid,
                                                          columns=['start_timestamp', 'end_timestamp', 'ctx_vector'])
                if uuid not in test_ids:
                    if X is None:
                        X = np.stack(x_uuid, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_uuid, axis=0)], axis=0)
                else:
                    logger.info(f"UUID {uuid} is a testing home, skipping it for training set")
                logger.info(f"Fetched data for id {uuid}.")

            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

            if access_cache:
                X_deduped = np.unique(X, axis=0)
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

            df_data = pd.pivot_table(df_data, index=['Home', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['Home', 'timestamp'])

            X = None
            context_request_dict = dict()

            for home in df_data['Home'].unique():
                df_home_data = df_data[df_data.Home == home]
                df_home_activities = df_home_data[args.onto_activity_labels]
                df_home_activities['activity_vec'] = df_home_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_home_activities.loc[:, 'timestamp'] = df_home_data['timestamp']
                df_home_activities = df_home_activities[['timestamp', 'activity_vec']]

                x_home = []
                ctx_req_home = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_home_activities.shape[0], args.sliding_parameter):
                    df_window = df_home_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length):
                        # x_home.append(np.concatenate(df_window['activity_vec'].values))
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_home.append(ctx_vector)
                        ctx_req_home.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[home] = pd.DataFrame(ctx_req_home,
                                                          columns=['start_timestamp', 'end_timestamp', 'ctx_vector'])
                if len(x_home) > 0:
                    if home not in test_ids:
                        if X is None:
                            X = np.stack(x_home, axis=0)
                        else:
                            X = np.concatenate([X, np.stack(x_home, axis=0)], axis=0)
                    else:
                        logger.info(f"Home {home} is a testing home, skipping it for training set")
                logger.info(f"Fetched data for id {home}.")
            X[X > 0] = 1.
            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

        if access_cache:
            X_deduped = np.unique(X, axis=0)
            pickle.dump((X, X_deduped, context_request_dict), open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X, X_deduped, context_request_dict


'''
This function parses realworld dataset and create a copy in cache
'''


def parse_realworld_dataset(rawdatasrc,
                            args=None,
                            cache_dir='cache/',
                            access_cache=False,
                            logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache
        lopo_user = rawdatasrc.split("/")[-1].split(".")[0]
        cache_file = f'{cache_dir}/realworld_{lopo_user}_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'

        if access_cache:
            logger.info(f"Trying to fetch data from {cache_file}")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got RealWorld Data from cache..")
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
            df_data = df_data[['session_id', 'timestamp', 'isTrain'] + args.activity_labels]
            df_data = pd.melt(df_data, id_vars=['session_id', 'timestamp', 'isTrain'], var_name='activity_name',
                              value_name='Activity')
            df_data = df_data[df_data.Activity == True]
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
                                              merge_mins * (x.minute // merge_mins), 0))
                df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            # convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(
                lambda x: ontolist.activity_mapping[args.dataset][x])

            df_data = pd.pivot_table(df_data, index=['session_id', 'timestamp', 'isTrain'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['session_id', 'timestamp'])

            X = None
            context_request_dict = dict()
            # do session_id level separation and create dataset for training
            for session_id in df_data['session_id'].unique():
                df_session_id_data = df_data[df_data.session_id == session_id]
                df_session_id_activities = df_session_id_data[args.onto_activity_labels]
                df_session_id_activities['activity_vec'] = df_session_id_activities.apply(
                    lambda row: np.array(row, dtype=float),
                    axis=1)
                df_session_id_activities.loc[:, 'timestamp'] = df_session_id_data['timestamp']
                df_session_id_activities.loc[:, 'isTrain'] = df_session_id_data['isTrain']
                df_session_id_activities = df_session_id_activities[['timestamp', 'activity_vec', 'isTrain']]

                x_session_id = []
                ctx_req_session_id = []
                window_length = int(args.lag_parameter / args.merge_mins)
                for row_idx in range(0, df_session_id_activities.shape[0], args.sliding_parameter):
                    df_window = df_session_id_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        is_train_window = False
                        if (df_window.isTrain.sum() > 1):
                            is_train_window = True
                            x_session_id.append(ctx_vector)
                        ctx_req_session_id.append(
                            [is_train_window, df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[session_id] = pd.DataFrame(ctx_req_session_id,
                                                                columns=['isTrain', 'start_timestamp', 'end_timestamp',
                                                                         'ctx_vector'])
                if (session_id not in test_ids) & (len(x_session_id) > 0):
                    if X is None:
                        X = np.stack(x_session_id, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_session_id, axis=0)], axis=0)
                else:
                    logger.info(f"Session {session_id} is a testing id, skipping it for training set")
                logger.info(f"Fetched data for id {session_id}.")

            t_datafetch_end = datetime.now()
            logger.info(f"Fetched data in {time_diff(t_datafetch_start, t_datafetch_end)} secs.")

            X_deduped = np.unique(X, axis=0)
            if access_cache:
                pickle.dump((X, X_deduped, context_request_dict), open(cache_file, 'wb'))
            else:
                ...

    return X, X_deduped, context_request_dict
