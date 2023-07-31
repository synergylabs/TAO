import numpy as np
import pandas as pd
from process_datasets import *
import datetime, time
import random
import os


# Get a dict of dataset processing functions
dataset_processor = {
    # 'extrasensory': process_extrasensory,
    'casas':process_casas,
}
dataset_files = {
    'extrasensory': 'datasets/extrasensory_dataset.csv',
    'casas':'datasets/casas_dataset.csv',
}
dataset_activities = {
    'casas':[
         'toilet','grooming','dressing_up','shower',
         'drinking','taking_meds','eating',
         'dishes_home','laundry_home',
         'reading','office_work',
         'lying','wakingup','sleeping',
         'meeting_friends',
         'cooking',
         'exercising',
         'watching_tv',
         'onphone',
         'step_in',
         'step_out',
    ]
}

result_dir = 'GT_marking/labelstudioinput'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def load_data(dataset_file, dataset):
    df_data = dataset_func(dataset_file)
    activity_list = df_data.activity.unique().tolist()
    # df_data['activity'] = df_data['activity'].apply(lambda x: activity_list.index(x))
    # df_data['datetime'] = pd.to_datetime(df_data['timestamp'], unit='s')
    df_data['date'] = df_data['datetime'].dt.date
    df_id_dates = df_data[['id', 'date']].drop_duplicates()
    df_data = df_data[['id', 'date', 'datetime', 'activity']]
    return df_data, df_id_dates, activity_list

def add_channel(activity, height = 40):
    channel_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    channel_string = f'<Channel column="{activity}" units="" height="{height}" displayFormat=",.0f" strokeColor="{channel_color}" legend="{activity}"/>'
    return channel_string


for dataset in dataset_processor.keys():
    dataset_func = dataset_processor[dataset]
    dataset_file = dataset_files[dataset]
    # load data
    df_data, df_id_dates, activity_list = load_data(dataset_file, dataset_func)

    for current_id in df_id_dates.id.unique():
        id_input_dir = f"{result_dir}/{current_id}"
        if not os.path.exists(id_input_dir):
            os.makedirs(id_input_dir)

        # setup file for given id
        df_id_data = df_data[(df_data.id == current_id)]
        id_activities = df_id_data['activity'].unique().tolist()
        id_sorted_activities =[]
        for activity in dataset_activities[dataset]:
            if activity in id_activities:
                id_sorted_activities.append(activity)

        channels = []
        for activity in id_sorted_activities:
            channels.append(add_channel(activity))
        with open(f"{id_input_dir}/channels.txt",'w') as f:
            f.writelines(channels)
        current_id_dates = df_id_dates[df_id_dates.id == current_id].date.unique().tolist()
        try:
            for current_date in current_id_dates:
                print(current_id,current_date)
                df_current_data = df_data[(df_data.id == current_id) & (df_data.date == current_date)]
                df_current_data['value'] = 1
                df_current_data = pd.pivot_table(df_current_data,index='datetime',columns='activity',values='value',aggfunc=np.sum)
                df_current_data = df_current_data.fillna(0.)
                date_activities = []
                not_date_activities = []
                for activity in id_activities:
                    if activity in df_current_data.columns:
                        date_activities.append(activity)
                        df_current_data[activity] = df_current_data[activity].values.astype(int)
                    else:
                        df_current_data[activity] = 0
                        not_date_activities.append(activity)

                df_current_data = df_current_data[date_activities+not_date_activities]
                df_current_data = df_current_data.reset_index()
                df_current_data['datetime'] = df_current_data['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
                df_current_data.to_csv(f"{id_input_dir}/{current_date}.csv",index=False)
        except:
            ...
    # break



