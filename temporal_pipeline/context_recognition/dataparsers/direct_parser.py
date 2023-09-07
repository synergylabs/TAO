'''
This file contains data parsers for all datasets with direct parsing
'''
import torch.utils.data
import numpy as np
import os
import pickle
import glob
import pandas as pd
from utils import time_diff, get_activity_vector
from datetime import datetime, timedelta

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

        cache_file = f'{cache_dir}/extrasensory_{args.lag_parameter}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'

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
            X = None
            X_labelled = []
            # do uuid level separation and create dataset for training
            for uuid in df_data['uuid'].unique():
                df_uuid_data = df_data[df_data.uuid == uuid]
                df_uuid_activities = df_uuid_data[args.activity_labels]
                df_uuid_activities['activity_vec'] = df_uuid_activities.apply(lambda row: np.array(row, dtype=float),
                                                                              axis=1)
                df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
                df_uuid_activities = df_uuid_activities[['timestamp', 'activity_vec']]

                # get gt labels info if available
                gt_dir = args.gt_label_dir
                df_uuid_labels = None
                if os.path.exists(f'{gt_dir}/{uuid}/'):
                    logger.debug(f"Collecting GT labels for uuid: {uuid}")
                    for filepath in glob.glob(f"{gt_dir}/{uuid}/*.txt"):
                        label_date = filepath.split("/")[-1].split(".")[0]  # todo: fixed parser, need to update
                        label_date = datetime.strptime(label_date, "%Y-%m-%d")
                        df_uuid_labels = pd.read_csv(filepath)
                        df_uuid_labels['start_time'] = df_uuid_labels['start_time'].apply(
                            lambda time_str: label_date + timedelta(
                                hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                        df_uuid_labels['start_time'] = df_uuid_labels['start_time'].dt.tz_localize(
                            'America/Los_Angeles').astype(np.int64) // 10 ** 9
                        df_uuid_labels['end_time'] = df_uuid_labels['end_time'].apply(
                            lambda time_str: label_date + timedelta(
                                hours=int(time_str.split(":")[0]), minutes=int(time_str.split(":")[1])))
                        df_uuid_labels['end_time'] = df_uuid_labels['end_time'].dt.tz_localize(
                            'America/Los_Angeles').astype(np.int64) // 10 ** 9

                x_uuid = []
                for row_idx in range(0, df_uuid_activities.shape[0], args.sliding_parameter):
                    df_window = df_uuid_activities.iloc[row_idx:row_idx + args.lag_parameter]
                    max_timestamp_diff = df_window['timestamp'].diff().max()
                    if (df_window.shape[0] == args.lag_parameter) & (
                            max_timestamp_diff < args.max_time_interval_in_mins * 60):
                        ctx_vector = np.concatenate(df_window['activity_vec'].values)
                        x_uuid.append(ctx_vector)
                        if df_uuid_labels is not None:
                            xt_min, xt_max = df_window.timestamp.min(), df_window.timestamp.max()
                            x_labels = df_uuid_labels[
                                (df_uuid_labels.start_time <= xt_min) & (df_uuid_labels.end_time >= xt_min) | (df_uuid_labels.start_time <= xt_max) & (df_uuid_labels.end_time >= xt_max)]
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
        X_labelled = None # not required for prediction, added for completeness
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
This function parses aruba dataset and create a copy in cache
'''


def parse_aruba_dataset(rawdatasrc,
                        args=None,
                        cache_dir='cache/',
                        access_cache=False,
                        logger=None):
    if args.mode == 'train':
        # check if cache dir is available, and try to read data from cache
        cache_file = f'{cache_dir}/aruba_{args.lag_parameter}_{args.sliding_parameter}_{args.max_time_interval_in_mins}.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Aruba Data from cache..")
                X = pickle.load(open(cache_file, 'rb'))
                return X

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.apply(
                lambda row: f"{row['activity_name'].strip()}", axis=1)
            df_data = pd.pivot_table(df_data, index='timestamp', columns='activity_name', values='activity',
                                     aggfunc='count')
            df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()

            # there is no uuid in data, just plugin all data directily
            df_activities = df_data[args.activity_labels]
            df_activities['activity_vec'] = df_activities.apply(lambda row: np.array(row, dtype=float),
                                                                axis=1)
            df_activities.loc[:, 'timestamp'] = df_data['timestamp']
            df_activities = df_activities[['timestamp', 'activity_vec']]
            # formatting from string to datetime
            df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'], format='%Y-%m-%d %H:%M:%S')
            # formatting from datetime to epoch secs
            df_activities['timestamp'] = pd.to_numeric(df_activities['timestamp'].values) / 10 ** 9

            x = []
            for row_idx in range(0, df_activities.shape[0], args.sliding_parameter):
                df_window = df_activities.iloc[row_idx:row_idx + args.lag_parameter]
                max_timestamp_diff = df_window['timestamp'].diff().max()
                if (df_window.shape[0] == args.lag_parameter) & (
                        max_timestamp_diff < args.max_time_interval_in_mins * 60):
                    x.append(np.concatenate(df_window['activity_vec'].values))
            X = np.stack(x, axis=0)

        if access_cache:
            pickle.dump(X, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return X


'''
This function parses witham dataset and create a copy in cache
'''


def parse_witham_dataset(rawdatasrc=None, args=None,
                         cache_dir='cache/activities',
                         access_cache=True):
    # check if cache dir is available, and try to read data from cache
    cache_file = f'{cache_dir}/witham.pickle'
    if access_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # check if file exists
        if os.path.exists(cache_file):
            print("Got Witham Data from cache..")
            X = pickle.load(open(cache_file, 'rb'))
            return X

    if rawdatasrc is None:
        raise FileNotFoundError
    else:
        df_data = pd.read_csv(rawdatasrc)
        X = None
        ...

    if access_cache:
        pickle.dump(X, open(cache_file, 'wb'))

    pass


'''
This function parses casas dataset and create a copy in cache
'''


def parse_casas_dataset(rawdatasrc=None, args=None,
                        cache_dir='cache/activities',
                        access_cache=True):
    # check if cache dir is available, and try to read data from cache
    cache_file = f'{cache_dir}/casas.pickle'
    if access_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # check if file exists
        if os.path.exists(cache_file):
            print("Got Casas Data from cache..")
            X = pickle.load(open(cache_file, 'rb'))
            return X

    if rawdatasrc is None:
        raise FileNotFoundError
    else:
        df_data = pd.read_csv(rawdatasrc)
        X = None
        ...

    if access_cache:
        pickle.dump(X, open(cache_file, 'wb'))

    pass


'''
This function parses opportunity dataset and create a copy in cache
'''


def parse_opportunity_dataset(rawdatasrc=None, args=None,
                              cache_dir='cache/activities',
                              access_cache=True):
    # check if cache dir is available, and try to read data from cache
    cache_file = f'{cache_dir}/opportunity.pickle'
    if access_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # check if file exists
        if os.path.exists(cache_file):
            print("Got Opportunity Data from cache..")
            X = pickle.load(open(cache_file, 'rb'))
            return X

    if rawdatasrc is None:
        raise FileNotFoundError
    else:
        df_data = pd.read_csv(rawdatasrc)
        X = None
        ...

    if access_cache:
        pickle.dump(X, open(cache_file, 'wb'))

    pass
