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
from wellness.stress_predictors.fsm_stress_pipeline import get_daily_stress_score

wellness_function_map = {
    'stress': get_daily_stress_score,
}


class wellnessPredictor:
    def __init__(self, run_config, logger):

        self.config = run_config
        self.logger = logger

        # create an active array to support data caching and buffering
        self.request_buffer_ts = []
        self.request_buffer_act = []
        # self.activity_label_count = len(self.config.onto_activity_labels)
        self.wellness_lag_parameter = run_config.wellness_lag_parameter
        self.max_history_length_in_mins = 7*24*60

    def process_request(self, request_json, type='stress'):
        for context in request_json['contexts']:
            self.request_buffer_ts.append(request_json['timestamp'])
            self.request_buffer_act.append(context)

        self.min_ts = min(self.request_buffer_ts)
        self.max_ts = max(self.request_buffer_ts)
        if self.max_ts - self.min_ts >= self.wellness_lag_parameter * 60:
            # drop older values
            eligible_idxs = \
                np.where(np.array(self.request_buffer_ts) > self.max_ts - (60 * self.max_history_length_in_mins))[0]
            self.request_buffer_ts = np.array(self.request_buffer_ts)[eligible_idxs].tolist()
            self.request_buffer_act = np.array(self.request_buffer_act)[eligible_idxs].tolist()
            if type == 'stress':
                response_string = self.get_stress_score()
            else:
                response_string = f'{type} score Not implemented'
                self.logger.info("Incomplete information to predict wellness")
        else:
            response_string = 'Unavailable'
            self.logger.info("Incomplete information to predict wellness")

        context_inputs = [
            f'{pd.to_datetime(self.request_buffer_ts[i], unit="s").strftime("%H:%M:%S")}:{self.request_buffer_act[i]}'
            for i in range(len(self.request_buffer_ts))]

        return response_string, context_inputs

    def get_stress_score(self):
        '''
        Get context vector from request buffer based on run config
        :param buffer:
        :return:
        '''
        lag_parameter = self.wellness_lag_parameter
        merge_mins = self.config.merge_mins
        df_wellness_input = pd.DataFrame(zip(self.request_buffer_ts, self.request_buffer_act),
                                         columns=['timestamp', 'context'])
        df_wellness_input['datetime'] = pd.to_datetime(df_wellness_input['timestamp'],
                                                       unit='s').dt.tz_localize('UTC').dt.tz_convert(
            'America/Los_Angeles').dt.tz_localize(None)

        df_user_input = None
        user_weeks = pd.to_datetime(df_wellness_input['timestamp'], unit='s').dt.tz_localize(
            'UTC').dt.tz_convert('America/Los_Angeles').dt.tz_localize(None).dt.strftime("%Y-%V").unique()


        user_week_context_attributes = []
        for week in user_weeks:
            week_context_attributes = {}
            df_wellness_input_week  = df_wellness_input[
                pd.to_datetime(df_wellness_input['timestamp'], unit='s').dt.tz_localize(
                    'UTC').dt.tz_convert('America/Los_Angeles').dt.tz_localize(None).dt.strftime("%Y-%V") == week]

            user_days = pd.to_datetime(df_wellness_input_week['timestamp'], unit='s').dt.tz_localize(
                'UTC').dt.tz_convert('America/Los_Angeles').dt.tz_localize(None).dt.strftime("%Y-%V_%Y-%m-%d").unique()
            for week_day in user_days:
                day_val = week_day.split("_")[1]
                df_day_wellness_input = df_wellness_input[
                    df_wellness_input['datetime'].dt.strftime("%Y-%m-%d") == day_val]
                df_day_wellness_input.columns = ['timestamp', 'context', 'datetime']
                df_ts_min, df_ts_max = df_day_wellness_input['timestamp'].min(), df_day_wellness_input['timestamp'].max()
                df_ts = pd.DataFrame(np.arange(df_ts_min, df_ts_max + 1, merge_mins*60), columns=['timestamp'])
                df_day_wellness_input = pd.merge(df_ts, df_day_wellness_input, on='timestamp', how='left')
                df_day_wellness_input['context'] = df_day_wellness_input['context'].fillna(method='ffill')
                df_day_wellness_input['datetime'] = pd.to_datetime(df_day_wellness_input['timestamp'],
                                                                    unit='s').dt.tz_localize('UTC').dt.tz_convert(
                        'America/Los_Angeles').dt.tz_localize(None)
                df_day_wellness_input['context_grp'] = (
                        df_day_wellness_input['context'] != df_day_wellness_input['context'].shift(1)).cumsum()
                day_contexts = df_day_wellness_input.groupby(['context_grp', 'context'], as_index=False).agg({
                    'timestamp': ['min', 'max', lambda x: x.max() - x.min()]
                })
                day_contexts.columns = ['group', 'context', 'start', 'end', 'length']
                week_context_attributes[day_val] = day_contexts

            user_week_context_attributes.append([week, week_context_attributes])
            weekly_stress_scores = []
            for week, week_context_attributes in user_week_context_attributes:
                week_stress_score = get_daily_stress_score(week_context_attributes)
                weekly_stress_scores.append((week, week_stress_score))
        return weekly_stress_scores
