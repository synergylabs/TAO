'''
This file contains parses context vectors for prediction based on stack and merge technique with compression parameter
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

        cache_file = f'{cache_dir}/extrasensory_predict_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'

        if access_cache:
            logger.info(f"Trying to fetch data from {cache_file}")
            if os.path.exists(cache_file):
                logger.info("Got Extrasensory Data from cache..")
                context_request_dict = pickle.load(open(cache_file, 'rb'))
                return context_request_dict
            else:
                logger.info(f"Cache file not available. Fetching from raw datafile")
        else:
            logger.info("Cache access not allowed. Fetching from raw datafile")

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data = df_data[['uuid', 'timestamp'] + args.activity_labels]
            df_data = pd.melt(df_data, id_vars=['uuid', 'timestamp'], var_name='activity_name', value_name='Activity')
            df_data = df_data[df_data.Activity == True]
            if not (args.merge_mins == -1):
                df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
                if args.merge_mins < 1:
                    merge_seconds = int(args.merge_mins * 60)
                    if(merge_seconds < 1):
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

            df_data['activity_name'] = df_data['activity_name'].apply(
                lambda x: ontolist.activity_mapping[args.dataset][x])
            df_data = pd.pivot_table(df_data, index=['uuid', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['uuid', 'timestamp'])

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
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_uuid_activities.shape[0], args.sliding_parameter):
                    df_window = df_uuid_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if df_window.shape[0] == window_length:
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_uuid.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])
                context_request_dict[uuid] = pd.DataFrame(x_uuid,
                                                          columns=['start_timestamp', 'end_timestamp', 'ctx_vector'])

            if access_cache:
                pickle.dump(context_request_dict, open(cache_file, 'wb'))
            else:
                ...

    return context_request_dict


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
        cache_file = f'{cache_dir}/casas_predict_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Casas Data from cache..")
                context_request_dict = pickle.load(open(cache_file, 'rb'))
                return context_request_dict

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
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
            #
            # df_data['timestamp'] = pd.to_datetime(df_data['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
            #
            # df_data['timestamp'] = df_data['timestamp'].apply(
            #     lambda x: pd.datetime(x.year, x.month, x.day, x.hour, args.merge_mins * (x.minute // args.merge_mins),
            #                           0))
            # df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            # df_data = pd.pivot_table(df_data, index=['Home', 'timestamp'], columns='activity_name', values='Activity',
            #                          aggfunc='count')
            # # df_data = df_data[args.activity_labels]
            # df_data = df_data.fillna(0.).reset_index()

            context_request_dict = dict()

            for home in df_data['Home'].unique():
                df_home_data = df_data[df_data.Home == home]
                df_home_activities = df_home_data[args.onto_activity_labels]
                df_home_activities['activity_vec'] = df_home_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_home_activities.loc[:, 'timestamp'] = df_home_data['timestamp']
                df_home_activities = df_home_activities[['timestamp', 'activity_vec']]

                x_home = []

                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_home_activities.shape[0], args.sliding_parameter):
                    df_window = df_home_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_home.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])

                if len(x_home) > 0:
                    context_request_dict[home] = pd.DataFrame(x_home, columns=['start_timestamp', 'end_timestamp',
                                                                               'ctx_vector'])
        if access_cache:
            pickle.dump(context_request_dict, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return context_request_dict


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
        cache_file = f'{cache_dir}/tsu_predict_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got TSU Data from cache..")
                context_request_dict = pickle.load(open(cache_file, 'rb'))
                return context_request_dict

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.Activity
            if not (args.merge_mins == -1):
                df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
                if args.merge_mins < 1:
                    merge_seconds = int(args.merge_mins * 60)
                    if(merge_seconds < 1):
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

            df_data = pd.pivot_table(df_data, index=['session_id', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['session_id', 'timestamp'])

            context_request_dict = dict()

            for session in df_data['session_id'].unique():
                df_session_data = df_data[df_data.session_id == session]
                df_session_activities = df_session_data[args.activity_labels]
                df_session_activities['activity_vec'] = df_session_activities.apply(
                    lambda row: np.array(row, dtype=float),
                    axis=1)
                df_session_activities.loc[:, 'timestamp'] = df_session_data['timestamp']
                df_session_activities = df_session_activities[['timestamp', 'activity_vec']]

                # get gt labels info if available
                x_session = []
                window_length = int(args.lag_parameter / args.merge_mins)
                for row_idx in range(0, df_session_activities.shape[0], args.sliding_parameter):
                    df_window = df_session_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if df_window.shape[0] == window_length:
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_session.append([df_window['timestamp'].min(), df_window['timestamp'].max(), ctx_vector])

                if len(x_session) > 0:
                    context_request_dict[session] = pd.DataFrame(x_session, columns=['start_timestamp', 'end_timestamp',
                                                                                     'ctx_vector'])
        if access_cache:
            pickle.dump(context_request_dict, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return context_request_dict


'''
This function parses opportunity dataset and create a copy in cache
'''


def parse_opportunity_dataset(rawdatasrc,
                              args=None,
                              cache_dir='cache/',
                              access_cache=False,
                              logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache
        cache_file = f'{cache_dir}/opportunity_stack_merge_{args.lag_parameter}_{args.merge_mins}_{args.sliding_parameter}_{args.max_time_interval_in_mins}_{args.merge_mins}.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Opportunity Data from cache..")
                X = pickle.load(open(cache_file, 'rb'))
                return X

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data = df_data[~(df_data.label_name == 'None')]
            df_data = df_data[~(df_data.observation == 'activity')]

            # remove Not Known
            df_data = df_data.rename(columns={'label_name': 'activity_name'})
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp_in_millisec'], unit='s')

            df_data['timestamp'] = df_data['timestamp'].apply(
                lambda x: pd.datetime(x.year, x.month, x.day, x.hour, args.merge_mins * (x.minute // args.merge_mins),
                                      0))
            df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            df_data = pd.pivot_table(df_data, index=['user', 'timestamp'], columns='activity_name', values='label',
                                     aggfunc='count')
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()

            # there is no uuid in data, just plugin all data directily

            X = None
            for user in df_data['user'].unique():
                df_user_data = df_data[df_data.user == user]
                df_user_activities = df_user_data[args.activity_labels]
                df_user_activities['activity_vec'] = df_user_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_user_activities.loc[:, 'timestamp'] = df_user_data['timestamp']
                df_user_activities = df_user_activities[['timestamp', 'activity_vec']]

                x_user = []
                for row_idx in range(0, df_user_activities.shape[0], args.sliding_parameter):
                    df_window = df_user_activities.iloc[row_idx:row_idx + args.lag_parameter]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == args.lag_parameter) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        x_user.append(np.concatenate(df_window['activity_vec'].values))
                if X is None:
                    X = np.stack(x_user, axis=0)
                else:
                    X = np.concatenate([X, np.stack(x_user, axis=0)], axis=0)

            X[X > 0] = 1.
        if access_cache:
            pickle.dump(X, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X
