'''
This is primary file to predict context based on learned deep clustering model from datasets
Author: Prasoon Patidar
Created At: Jul 22, 2022
'''

# core libraries
import sys
import os
import logging
from datetime import datetime
from logging.handlers import WatchedFileHandler
import traceback
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import csv
import argparse
from collections import namedtuple
import pickle
import numpy as np
import jstyleson as json
from flask import Flask, request
import time
from queue import Queue
from copy import deepcopy
# custom libraries
from utils import time_diff, get_config_from_json
from context_recognition.contextPredictor import contextPredictor
from context_recognition.dataloaders import load_data_parser
from context_recognition import fetch_prediction_requests
from wellness.wellnessPredictor import wellnessPredictor



def init_context_predictor(config, base_config='context_configs/default.json', logger=None):
    '''
    This function loads the models and setup server for context prediction
    :param config: Main config to setup server
    :param base_config: Default file with complete set of config requirements
    :return: server init status: success/failure
    '''

    # Load configuration and requested modelsx

    # Get config dict from default and config parameters

    run_config = get_config_from_json(base_config)
    run_config.update(config)
    if 'input_size' not in run_config.keys():
        run_config['input_size'] = int(run_config['lag_parameter'] / run_config['merge_mins'])
        run_config['input_size'] = run_config['input_size'] * len(run_config['onto_activity_labels'])
    run_config['data_sample'] = np.zeros(run_config['input_size'])
    run_config = namedtuple('run_config', run_config.keys())(*run_config.values())

    if not os.path.exists(run_config.cache_dir):
        os.makedirs(run_config.cache_dir)

    # Initialize context predictor
    context_predictor = contextPredictor(run_config, logger)

    return context_predictor, run_config


def initialize_logger(log_dir):
    logger_master = logging.getLogger('context_prediction')
    logger_master.setLevel(logging.DEBUG)
    ## Add core logger handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(f"{log_dir}/prediction.log")
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    ## Add stdout logger handler
    console_formatter = logging.Formatter(
        '%(asctime)s | %(module)s | %(levelname)s | %(message)s')
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(console_formatter)
    logger_master.addHandler(console_log)
    logger = logging.LoggerAdapter(logger_master, {})
    return logger


def init_extrasensory_predictor(global_config, cluster_counts, input_sizes, logger):
    lag_parameter = global_config["model_configs"]["lag_window"]
    dconfig = {'dataset': 'extrasensory',
               'base_config': 'context_configs/extrasensory.json',
               'lag_parameter': lag_parameter,
               'merge_mins': global_config["model_configs"]["stack_window"],
               'max_time_interval_in_mins': global_config["model_configs"]["max_time_interval"],
               'input_size': input_sizes['extrasensory'][str(lag_parameter)],
               'cnet_n_clusters': cluster_counts['extrasensory'][str(lag_parameter)]}

    mconfig = {'model_re': 'TAE', 'stacked_input': False}

    # create config for predictor

    experiment_name = f"{dconfig['dataset']}_{dconfig['lag_parameter']}_"
    experiment_name += f"{dconfig['merge_mins']}_{mconfig['model_re']}"
    new_config = deepcopy(global_config)
    new_config.update(dconfig)
    new_config.update(mconfig)
    new_config['device'] = 'cpu'
    new_config['experiment'] = experiment_name
    new_config['cache_dir'] = new_config['models_dir']
    base_config = new_config['base_config']

    # Initialize predictor
    context_predictor, run_config = init_context_predictor(new_config, base_config, logger)

    return context_predictor, run_config


def init_casas_predictor(global_config, cluster_counts, input_sizes, logger):
    lag_parameter = global_config["model_configs"]["lag_window"]
    dconfig = {'dataset': 'casas',
               'base_config': 'context_configs/casas.json',
               'lag_parameter': lag_parameter,
               'merge_mins': global_config["model_configs"]["stack_window"],
               'max_time_interval_in_mins': global_config["model_configs"]["max_time_interval"],
               'input_size': input_sizes['casas'][str(lag_parameter)],
               'cnet_n_clusters': cluster_counts['casas'][str(lag_parameter)]}
    mconfig = {'model_re': 'TAE', 'stacked_input': False}

    # create config for predictor

    experiment_name = f"{dconfig['dataset']}_{dconfig['lag_parameter']}_"
    experiment_name += f"{dconfig['merge_mins']}_{mconfig['model_re']}"
    new_config = deepcopy(global_config)
    new_config.update(dconfig)
    new_config.update(mconfig)
    new_config['device'] = 'cpu'
    new_config['experiment'] = experiment_name
    new_config['cache_dir'] = new_config['models_dir']
    base_config = new_config['base_config']

    # Initialize predictor
    context_predictor, run_config = init_context_predictor(new_config, base_config, logger)

    return context_predictor, run_config


def init_wellness_predictor(global_config, logger):
    wellness_lag_parameter = global_config["model_configs"]["wellness_lag_window"]
    new_config = {'wellness_lag_parameter': wellness_lag_parameter,
               'merge_mins': global_config["model_configs"]["stack_window"],
               'max_time_interval_in_mins': global_config["model_configs"]["max_time_interval"]}

    # Initialize stress
    new_config = namedtuple('new_config', new_config.keys())(*new_config.values())
    wellness_predictor = wellnessPredictor(new_config, logger)

    return wellness_predictor, new_config
