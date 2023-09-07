'''
This file contains parsing method to replicate (timestamp, activity) requests architecture for prediction service from
all datasets. This parser doesn't concern with create context vectors on its own.
Author: Prasoon Patidar
Created at: 24th Jul 2022
'''
import torch.utils.data
import numpy as np
import os
import pickle
import pandas as pd
import glob
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
    # check if cache dir is available, and try to read data from cache

    cache_file = f'{cache_dir}/extrasensory_prediction_requests.pb'

    if access_cache:
        logger.info(f"Trying to fetch data from {cache_file}")
        if os.path.exists(cache_file):
            logger.info("Got Extrasensory Data from cache..")
            context_detection_requests_dict = pickle.load(open(cache_file, 'rb'))
            return context_detection_requests_dict
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
        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
        df_data = pd.pivot_table(df_data, index=['uuid', 'timestamp'], columns='activity_name',
                                 values='Activity', aggfunc='count')
        df_data[df_data > 1.] = 1.
        # df_data = df_data[args.activity_labels]
        df_data = df_data.fillna(0.).reset_index()
        df_data = df_data.sort_values(by=['uuid', 'timestamp'])
        df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
        context_detection_requests_dict = dict()
        # do uuid level separation and create dataset for training
        for uuid in df_data['uuid'].unique():
            df_uuid_data = df_data[df_data.uuid == uuid]
            df_uuid_activities = df_uuid_data[args.activity_labels]
            df_uuid_activities['activities'] = df_uuid_activities.apply(
                lambda row: ','.join([col for col in row.keys() if float(row[col] == 1.)]), axis=1)
            df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
            df_uuid_activities = df_uuid_activities[['timestamp', 'activities']].sort_values(by='timestamp')
            context_detection_requests_dict[uuid] = df_uuid_activities

        if access_cache:
            pickle.dump(context_detection_requests_dict, open(cache_file, 'wb'))
        else:
            ...

    return context_detection_requests_dict


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
        cache_file = f'{cache_dir}/casas_prediction_requests.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got Casas Data from cache..")
                context_detection_requests_dict = pickle.load(open(cache_file, 'rb'))
                return context_detection_requests_dict

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.Activity.apply(
                lambda x: x.strip().replace('r1.', '').replace('r2.', ''))
            df_data['timestamp'] = pd.to_datetime(df_data['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
            df_data = pd.pivot_table(df_data, index=['Home', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['Home', 'timestamp'])
            df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            context_detection_requests_dict = dict()
            for home in df_data['Home'].unique():
                df_home_data = df_data[df_data.Home == home]
                df_home_activities = df_home_data[args.activity_labels]
                df_home_activities['activities'] = df_home_activities.apply(
                    lambda row: ','.join([col for col in row.keys() if float(row[col] == 1.)]), axis=1)
                df_home_activities.loc[:, 'timestamp'] = df_home_data['timestamp']
                df_home_activities = df_home_activities[['timestamp', 'activities']].sort_values(by='timestamp')
                context_detection_requests_dict[home] = df_home_activities

        if access_cache:
            pickle.dump(context_detection_requests_dict, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return context_detection_requests_dict


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
        cache_file = f'{cache_dir}/tsu_prediction_requests.pb'
        if access_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            # check if file exists
            if os.path.exists(cache_file):
                logger.info("Got TSU Data from cache..")
                context_detection_requests_dict = pickle.load(open(cache_file, 'rb'))
                return context_detection_requests_dict

        if rawdatasrc is None:
            raise FileNotFoundError
        else:
            df_data = pd.read_csv(rawdatasrc)
            df_data['activity_name'] = df_data.Activity
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='s')
            df_data = pd.pivot_table(df_data, index=['session_id', 'timestamp'], columns='activity_name',
                                     values='Activity', aggfunc='count')
            df_data[df_data > 1.] = 1.
            # df_data = df_data[args.activity_labels]
            df_data = df_data.fillna(0.).reset_index()
            df_data = df_data.sort_values(by=['session_id', 'timestamp'])
            df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
            context_detection_requests_dict = dict()

            for session in df_data['session_id'].unique():
                df_session_data = df_data[df_data.session_id == session]
                df_session_activities = df_session_data[args.activity_labels]
                df_session_activities['activities'] = df_session_activities.apply(
                    lambda row: ','.join([col for col in row.keys() if float(row[col] == 1.)]), axis=1)
                df_session_activities.loc[:, 'timestamp'] = df_session_data['timestamp']
                df_session_activities = df_session_activities[['timestamp', 'activities']].sort_values(by='timestamp')
                context_detection_requests_dict[session] = df_session_activities

        if access_cache:
            pickle.dump(context_detection_requests_dict, open(cache_file, 'wb'))
    else:  # arg mode is predict and we are parsing a stream of timestamp datasets
        ...

    return context_detection_requests_dict
