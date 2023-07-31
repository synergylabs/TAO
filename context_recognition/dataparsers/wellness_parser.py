'''
This parser parses datasets assuming we wish to run a wellness module on them, it uses raw data file,
and caches data separately from training data
'''
import numpy as np
import os
import pickle
import pandas as pd
from utils import time_diff, get_activity_vector


def parse_extrasensory_dataset(rawdatasrc,
                               args=None,
                               cache_dir='cache/',
                               access_cache=False,
                               logger=None):

    # check if cache dir is available, and try to read data from cache

    cache_file = f'{cache_dir}/extrasensory_wellness_parsed_data.pb'

    if access_cache:
        logger.info(f"Trying to fetch data from {cache_file}")
        if os.path.exists(cache_file):
            logger.info("Got Extrasensory Data from cache..")
            df_parsed_data = pickle.load(open(cache_file,'rb'))
            return df_parsed_data
        else:
            logger.info(f"Cache file not available. Fetching from raw datafile")
    else:
        logger.info("Cache access not allowed. Fetching from raw datafile")

    if rawdatasrc is None:
        raise FileNotFoundError
    else:
        df_data = pd.read_csv(rawdatasrc)
        uuid_dfs = []
        start_times, end_times, ids = [],[],[]
        # do uuid level separation and create dataset for training
        for uuid in df_data['uuid'].unique():
            df_uuid_data = df_data[df_data.uuid == uuid]
            df_uuid_activities = df_uuid_data[args.activity_labels]
            df_uuid_activities['activity_vec'] = df_uuid_activities.apply(lambda row: np.array(row, dtype=float),
                                                                          axis=1)
            df_uuid_activities.loc[:, 'timestamp'] = df_uuid_data['timestamp']
            df_uuid_activities.loc[:, 'id'] = uuid
            df_uuid_activities = df_uuid_activities[['id','timestamp', 'activity_vec']]
            uuid_dfs.append(df_uuid_activities)

        df_parsed_data = pd.concat(uuid_dfs,ignore_index=True)

    if access_cache:
        pickle.dump(df_parsed_data, open(cache_file,'wb'))

    return df_parsed_data


'''
This function parses aruba dataset and create a copy in cache
'''
def parse_aruba_dataset(rawdatasrc,
                        args=None,
                        cache_dir='cache/',
                        access_cache=False,
                        logger=None):


    # check if cache dir is available, and try to read data from cache
    cache_file = f'{cache_dir}/aruba_wellness_parsed_data.pb'
    if access_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # check if file exists
        if os.path.exists(cache_file):
            logger.info("Got Aruba Data from cache..")
            df_activities = pickle.load(open(cache_file,'rb'))
            return df_activities

    if rawdatasrc is None:
        raise FileNotFoundError
    else:
        df_data = pd.read_csv(rawdatasrc)
        df_data['activity_name'] = df_data.apply(
            lambda row: f"{row['activity_name'].strip()}", axis=1)
        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], format='%Y-%m-%d %H:%M:%S')

        df_data['timestamp'] = df_data['timestamp'].apply(lambda x: pd.datetime(x.year, x.month, x.day, x.hour, args.merge_mins * (x.minute // args.merge_mins), 0))
        df_data['timestamp'] = pd.to_numeric(df_data['timestamp'].values) / 10 ** 9
        df_data = pd.pivot_table(df_data,index='timestamp',columns='activity_name',values='activity',aggfunc='count')
        df_data = df_data[args.activity_labels]
        df_data = df_data.fillna(0.).reset_index()
        # there is no uuid in data, just plugin all data directily
        df_activities = df_data[args.activity_labels]
        df_activities['activity_vec'] = df_activities.apply(lambda row: np.array(row, dtype=float),
                                                                      axis=1)
        df_activities.loc[:, 'timestamp'] = df_data['timestamp']
        df_activities.loc[:, 'id'] = 'None'

        df_activities = df_activities[['id','timestamp', 'activity_vec']]
        df_activities['timestamp'] = pd.to_numeric(df_activities['timestamp'].values) / 10 ** 9

    if access_cache:
        pickle.dump(df_activities, open(cache_file,'wb'))

    return df_activities