'''
This file contains Dataset loaders for activity datasets
'''
import torch.utils.data
import numpy as np
import os
import pickle
import pandas as pd
from utils import time_diff, get_activity_vector
from datetime import datetime

from context_recognition.dataparsers import direct_parser, onto_conv_parser, ontoconv_prediction, combined_parser, incremental_parser

pd.set_option('mode.chained_assignment', None)


def load_data_parser(dataset, parse_style, logger):
    '''
    Get parser based on dataset and parsing style
    :param dataset: the activity dataset we are talking about
    :param parse_style: style of data to parse
    :return:
    '''

    # get relevant parsing style
    parser_func = None
    if parse_style == 'direct':
        parser_func = direct_parser
    # elif parse_style == 'stack_merge':
    #     parser_func = stack_merge_parser
    elif parse_style == 'onto_conv':
        parser_func = onto_conv_parser
    # elif parse_style == 'wellness':
    #     parser_func = wellness_parser
    elif parse_style == 'incremental':
        parser_func = incremental_parser
    # elif parse_style == 'prediction':
    #     parser_func = prediction_parser
    # elif parse_style == 'prediction_vector':
    #     parser_func = prediction_vector_parser
    elif parse_style == 'ontoconv_prediction':
        parser_func = ontoconv_prediction
    elif parse_style == 'combined':
        parser_func = combined_parser

    # get relevant dataset function

    data_parse_func = None
    if dataset == 'extrasensory':
        data_parse_func = parser_func.parse_extrasensory_dataset
    if dataset == 'aruba':
        data_parse_func = parser_func.parse_aruba_dataset
    if dataset == 'witham':
        data_parse_func = parser_func.parse_witham_dataset
    if dataset == 'casas':
        data_parse_func = parser_func.parse_casas_dataset
    if dataset == 'opportunity':
        data_parse_func = parser_func.parse_opportunity_dataset
    if dataset == 'realworld':
        data_parse_func = parser_func.parse_realworld_dataset

    if data_parse_func is None:
        logger.info(f"Unable to get data parser for dataset {dataset} and parsing style {parse_style}. Exiting...")
        exit(1)

    return data_parse_func


class activityDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        super(activityDataset, self).__init__()
        self.x = torch.from_numpy(X.astype(np.float32))

    def __getitem__(self, index):
        sample = self.x[index]
        return sample

    def __len__(self):
        return self.x.shape[0]
