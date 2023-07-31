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
            if os.path.exists(cache_file):
                logger.info("Got Extrasensory Data from cache..")
                X, X_deduped, X_labelled = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, X_labelled
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
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute, merge_seconds * (
                                x.second // merge_seconds)))
                else:
                    merge_mins = int(args.merge_mins)
                    df_data['timestamp'] = df_data['timestamp'].apply(
                        lambda x: pd.datetime(x.year, x.month, x.day, x.hour,
                                              merge_mins * (x.minute // merge_mins), 0))
                df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            #convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(lambda x: ontolist.activity_mapping[args.dataset][x])

            df_data = pd.pivot_table(df_data, index=['uuid', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['uuid', 'timestamp'])

            X = None
            X_labelled = []
            # do uuid level separation and create dataset for training
            for uuid in df_data['uuid'].unique():
                df_uuid_data = df_data[df_data.uuid == uuid]
                df_uuid_activities = df_uuid_data[args.onto_activity_labels]
                df_uuid_activities['activity_vec'] = df_uuid_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
                df_uuid_activities = df_uuid_activities[['timestamp', 'activity_vec']]

                # get gt labels info if available
                gt_dir = args.gt_label_dir
                df_uuid_labels = None
                # if os.path.exists(f'{gt_dir}/{uuid}/'):
                #     logger.debug(f"Collecting GT labels for uuid: {uuid}")
                #     for filepath in glob.glob(f"{gt_dir}/{uuid}/*.txt"):
                #         label_date = filepath.split("/")[-1].split(".")[0]  # todo: fixed parser, need to update
                #         label_date = datetime.strptime(label_date, "%Y-%m-%d")
                #         df_uuid_labels = pd.read_csv(filepath)
                #         df_uuid_labels['start_time'] = df_uuid_labels['start_time'].apply(
                #             lambda time_str: label_date + timedelta(
                #                 hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                #         df_uuid_labels['start_time'] = df_uuid_labels['start_time'].dt.tz_localize(
                #             'America/Los_Angeles').astype(np.int64) // 10 ** 9
                #         df_uuid_labels['end_time'] = df_uuid_labels['end_time'].apply(
                #             lambda time_str: label_date + timedelta(
                #                 hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                #         df_uuid_labels['end_time'] = df_uuid_labels['end_time'].dt.tz_localize(
                #             'America/Los_Angeles').astype(np.int64) // 10 ** 9

                x_uuid = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_uuid_activities.shape[0], args.sliding_parameter):
                    df_window = df_uuid_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_uuid.append(ctx_vector)
                        if df_uuid_labels is not None:
                            xt_min, xt_max = df_window.timestamp.min(), df_window.timestamp.max()
                            x_labels = df_uuid_labels[
                                (df_uuid_labels.start_time <= xt_min) & (df_uuid_labels.end_time >= xt_min) | (
                                        df_uuid_labels.start_time <= xt_max) & (df_uuid_labels.end_time >= xt_max)]
                            if x_labels.shape[0] > 0:
                                X_labelled.append((ctx_vector, x_labels.values.tolist()))

                if X is None:
                    X = np.stack(x_uuid, axis=0)
                else:
                    X = np.concatenate([X, np.stack(x_uuid, axis=0)], axis=0)
            if access_cache:
                X_deduped = np.unique(X, axis=0)
                pickle.dump((X, X_deduped, X_labelled), open(cache_file, 'wb'))
            else:
                ...

    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        X = None
        X_labelled = None  # not required for prediction, added for completeness
        data_arr = []
        for request_object in rawdatasrc:
            ts = request_object['timestamp']
        activities = request_object['activities']
        activity_vec = get_activity_vector(activities, args.activity_labels)
        data_arr.append([ts, activity_vec])
        df_activities = pd.DataFrame(data_arr, columns=['timestamp', 'activity_vec'])
        if df_activities.shape[0] < args.lag_parameter:
            logger.error("Unable to fetch enough datapoints to create context input vector..")
        elif df_activities.shape[0] == args.lag_parameter:
            X = np.concatenate(df_activities['activity_vec'].values)
        else:
            for row_idx in range(0, df_activities.shape[0], args.sliding_parameter):
                df_window = df_activities.iloc[row_idx:row_idx + args.lag_parameter]
                max_timestamp_diff = df_window['timestamp'].diff().max()
                if (df_window.shape[0] == args.lag_parameter) & (
                        max_timestamp_diff < args.max_time_interval_in_mins * 60):
                    data_arr.append(np.concatenate(df_window['activity_vec'].values))
            X = np.stack(data_arr, axis=0)

    return X, X_deduped, X_labelled


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
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Casas Data from cache..")
                X, X_deduped, X_labelled = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, X_labelled

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

            #convert activity name to that of ontology map
            df_data['activity_name'] = df_data['activity_name'].apply(lambda x: ontolist.activity_mapping[args.dataset][x])

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

            X = None
            X_labelled = []

            for home in df_data['Home'].unique():
                df_home_data = df_data[df_data.Home == home]
                df_home_activities = df_home_data[args.onto_activity_labels]
                df_home_activities['activity_vec'] = df_home_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_home_activities.loc[:, 'timestamp'] = df_home_data['timestamp']
                df_home_activities = df_home_activities[['timestamp', 'activity_vec']]

                # get gt labels info if available
                gt_dir = args.gt_label_dir
                df_home_labels = None
                # if os.path.exists(f'{gt_dir}/{home}/'):
                #     logger.debug(f"Collecting GT labels for home: {home}")
                #     for filepath in glob.glob(f"{gt_dir}/{home}/*.txt"):
                #         label_date = filepath.split("/")[-1].split(".")[0]  # todo: fixed parser, need to update
                #         label_date = datetime.strptime(label_date, "%Y-%m-%d")
                #         df_home_labels = pd.read_csv(filepath)
                #         df_home_labels['start_time'] = df_home_labels['start_time'].apply(
                #             lambda time_str: label_date + timedelta(
                #                 hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                #         df_home_labels['start_time'] = df_home_labels['start_time'].astype(np.int64) // 10 ** 9
                #         df_home_labels['end_time'] = df_home_labels['end_time'].apply(
                #             lambda time_str: label_date + timedelta(
                #                 hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                #         df_home_labels['end_time'] = df_home_labels['end_time'].astype(np.int64) // 10 ** 9
                #         df_home_labels['context'] = df_home_labels['context'].apply(
                #             lambda x: 'PreparingMeal' if (x == 'Meal_Preparation') else
                #             'PhoneCall' if (x == 'OnaPhoneCall') else x
                #         )

                x_home = []
                window_length = args.lag_parameter // args.merge_mins
                for row_idx in range(0, df_home_activities.shape[0], args.sliding_parameter):
                    df_window = df_home_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        # x_home.append(np.concatenate(df_window['activity_vec'].values))
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_home.append(ctx_vector)
                        if df_home_labels is not None:
                            xt_min, xt_max = df_window.timestamp.min(), df_window.timestamp.max()
                            x_labels = df_home_labels[
                                (df_home_labels.start_time <= xt_min) & (df_home_labels.end_time >= xt_min) | (
                                        df_home_labels.start_time <= xt_max) & (df_home_labels.end_time >= xt_max)]
                            if x_labels.shape[0] > 0:
                                X_labelled.append((ctx_vector, x_labels.values.tolist()))
                if len(x_home) > 0:
                    if X is None:
                        X = np.stack(x_home, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_home, axis=0)], axis=0)

            X[X > 0] = 1.
        if access_cache:
            X_deduped = np.unique(X, axis=0)
            pickle.dump((X, X_deduped, X_labelled), open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X, X_deduped, X_labelled


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
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got TSU Data from cache..")
                X, X_deduped, X_labelled = pickle.load(open(cache_file, 'rb'))
                return X, X_deduped, X_labelled

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
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

            df_data = pd.pivot_table(df_data, index=['session_id', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['session_id', 'timestamp'])

            X = None
            X_labelled = []
            df_tsu_gt_labels = None
            if os.path.exists(args.gt_label_dir):
                df_tsu_gt_labels = pd.read_csv(args.gt_label_dir)

            for session in df_data['session_id'].unique():
                df_session_data = df_data[df_data.session_id == session]
                df_session_activities = df_session_data[args.activity_labels]
                df_session_activities['activity_vec'] = df_session_activities.apply(
                    lambda row: np.array(row, dtype=float),
                    axis=1)
                df_session_activities.loc[:, 'timestamp'] = df_session_data['timestamp']
                df_session_activities = df_session_activities[['timestamp', 'activity_vec']]

                # get gt labels info if available
                df_session_labels = None
                if df_tsu_gt_labels is not None:
                    df_labels = df_tsu_gt_labels[df_tsu_gt_labels['session_id'] == session]
                    if df_labels.shape[0] > 0:
                        logger.info(f"Got labels for session {session}")
                        df_session_labels = df_labels[['start_time', 'end_time', 'context']]

                x_session = []
                window_length = int(args.lag_parameter / args.merge_mins)
                for row_idx in range(0, df_session_activities.shape[0], args.sliding_parameter):
                    df_window = df_session_activities.iloc[row_idx:row_idx + window_length]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == window_length) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        # x_session.append(np.concatenate(df_window['activity_vec'].values))
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_session.append(ctx_vector)
                        if df_session_labels is not None:
                            xt_min, xt_max = df_window.timestamp.min(), df_window.timestamp.max()
                            x_labels = df_session_labels[
                                (df_session_labels.start_time <= xt_min) & (df_session_labels.end_time >= xt_min) | (
                                        df_session_labels.start_time <= xt_max) & (
                                        df_session_labels.end_time >= xt_max)]
                            if x_labels.shape[0] > 0:
                                X_labelled.append((ctx_vector, x_labels.values.tolist()))
                if len(x_session) > 0:
                    if X is None:
                        X = np.stack(x_session, axis=0)
                    else:
                        X = np.concatenate([X, np.stack(x_session, axis=0)], axis=0)

            X[X > 0] = 1.
        if access_cache:
            X_deduped = np.unique(X, axis=0)
            pickle.dump((X, X_deduped, X_labelled), open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X, X_deduped, X_labelled


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
