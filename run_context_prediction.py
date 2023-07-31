'''
This is primary file to predict context based on learned deep clustering model from datasets
Author: Prasoon Patidar
Created At: Jul 22, 2022
'''

'''
This is main flask service to run context predictions based on configs
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
from flask import Flask, request
import time
from queue import Queue
from copy import deepcopy
# custom libraries
from utils import time_diff, get_config_from_json
from context_recognition.contextPredictor import contextPredictor
from context_recognition.dataloaders import load_data_parser
from context_recognition import fetch_prediction_requests


# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


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
        run_config['input_size'] = run_config['input_size'] * len(run_config['activity_labels'])
    run_config['data_sample'] = np.zeros(run_config['input_size'])
    run_config = namedtuple('run_config', run_config.keys())(*run_config.values())

    if not os.path.exists(run_config.cache_dir):
        os.makedirs(run_config.cache_dir)

    # Initialize context predictor
    context_predictor = contextPredictor(run_config, logger)

    return context_predictor, run_config


# @app.route('/predict', methods=['GET'])
# def process_request():
#     ts = request.args.get('timestamp')
#     activities = request.args.get('activities')
#     request_dict = {
#         'timestamp': ts,
#         'activities': activities.split(',')
#     }
#     context_label = context_predictor.process_request(request_dict)
#
#     response_dict = {ts: context_label}
#     return response_dict

def process_request(context_predictor, ts, activities):
    # ts = request.args.get('timestamp')
    # activities = request.args.get('activities')
    request_dict = {
        'timestamp': ts,
        'activities': activities.split(',')
    }
    context_label = context_predictor.process_request(request_dict)

    response_dict = {ts: context_label}
    return response_dict


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


if __name__ == '__main__':

    run_mode = 'multiple'

    if run_mode == 'multiple':
        # from train_configs.jul22_configs import get_configs
        # from train_configs.jul26_custom_configs import get_configs
        from train_configs.jul24_incremental_configs import get_configs
        # from train_configs.jul28_ontoconv_config import get_configs
        # prediction_parser = 'prediction_vector'
        prediction_parser = 'ontoconv_prediction'

        final_configs = get_configs()
        log_dir = final_configs[0]['log_dir']
        rlogger = initialize_logger(log_dir)
        for rconfig in final_configs:
            rbase_config = rconfig['base_config']
            try:
                new_config = deepcopy(rconfig)
                # new_config['device'] = 'cuda:0'
                base_config = new_config['base_config']
                # Initialize predictor
                context_predictor, run_config = init_context_predictor(new_config, base_config, rlogger)
                get_requests_func = load_data_parser(new_config['dataset'], prediction_parser, rlogger)
                context_requests_dict = fetch_prediction_requests(run_config, prediction_parser, rlogger)
                context_responses_dict = dict()
                rlogger.info(f"Got context requests dict with {len(context_requests_dict.keys())} unique ids")
                for id_idx, id in enumerate(context_requests_dict.keys()):
                    context_responses_id = []
                    df_requests = context_requests_dict[id]
                    rlogger.info(f"parsing request for id {id_idx}: {id} with {df_requests.shape[0]} context vectors")

                    for row_idx, row in df_requests.iterrows():
                        # print(len(row['ctx_vector']))
                        start_timestamp, end_timestamp, ctx_vector = row['start_timestamp'], row['end_timestamp'], \
                                                                     row['ctx_vector']
                        try:
                            row_context_response = context_predictor.predict(ctx_vector)
                            context_responses_id.append([start_timestamp, end_timestamp, row_context_response])
                        except KeyboardInterrupt:
                            sys.exit(1)
                        except:
                            rlogger.info(
                                f"Error for {int(start_timestamp)}, {int(end_timestamp)}, {str(ctx_vector)}")
                    context_responses_dict[id] = pd.DataFrame(context_responses_id,
                                                              columns=['start_timestamp', 'end_timestamp',
                                                                       'context'])
                    # logger.info("Empty context cache to process new user")
                    # context_predictor.empty_buffer()

                time_str = datetime.now().strftime("%Y%m%d")
                context_responses_filepath = f"{run_config.cache_dir}/{run_config.experiment}/context_responses_{time_str}.pb"
                pickle.dump(context_responses_dict, open(context_responses_filepath, 'wb'))
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                rlogger.info(f"Error in running experiment {rconfig['experiment']}, {str(traceback.format_exc())}")
                with open(f'{log_dir}/error_experiments.log', 'a+') as f:
                    f.write(f"{rconfig['experiment']},{str(traceback.format_exc())}\n")

    elif run_mode == 'single':
        # put data and model config for predictor
        dconfig = {'dataset': 'tsu', 'base_config': 'context_configs/tsu.json',
                   'lag_parameter': 0.5, 'merge_mins': 0.05, 'max_time_interval_in_mins': 1}
        mconfig = {'model_re': 'FCN', 'stacked_input': False, 'input_size': 414}

        # get global config
        from train_configs.jul22_configs import global_config

        # Initialize the logger
        log_dir = 'cache/logs/prediction'
        logger = initialize_logger(log_dir)

        # create config for predictor

        experiment_name = f"{dconfig['dataset']}_{dconfig['lag_parameter']}_"
        experiment_name += f"{dconfig['merge_mins']}_{mconfig['model_re']}"
        new_config = deepcopy(global_config)
        new_config.update(dconfig)
        new_config.update(mconfig)
        new_config['device'] = 'cuda:0'
        new_config['experiment'] = experiment_name
        base_config = new_config['base_config']

        # Initialize predictor
        context_predictor, run_config = init_context_predictor(new_config, base_config, logger)

        # app.run('0.0.0.0', port=8080, debug=True)

        # get requests based on dataset
        prediction_parser = 'prediction'
        get_requests_func = load_data_parser(new_config['dataset'], prediction_parser, logger)
        context_requests_dict = fetch_prediction_requests(run_config, prediction_parser, logger)
        sys.exit(0)
        context_responses_dict = dict()
        logger.info(f"Got context requests dict with {len(context_requests_dict.keys())} unique ids")
        for id in context_requests_dict.keys():
            if prediction_parser == 'prediction':
                context_responses_id = []
                df_requests = context_requests_dict[id]
                logger.info(f"parsing request for id: {id} with {df_requests.shape[0]} timestamp values")
                for row_idx, row in df_requests.iterrows():
                    timestamp, activities = row['timestamp'], row['activities']
                    try:
                        row_context_response = process_request(context_predictor, timestamp, activities)
                        context_responses_id.append([timestamp, activities, row_context_response[timestamp]])
                    except:
                        logger.info(f"Error for {int(timestamp)}, {activities}")
                context_responses_dict[id] = pd.DataFrame(context_responses_id,
                                                          columns=['timestamp', 'activities', 'context'])
                logger.info("Empty context cache to process new user")
                context_predictor.empty_buffer()
            else:  # prediction parser is prediction_vector
                context_responses_id = []
                df_requests = context_requests_dict[id]
                logger.info(f"parsing request for id: {id} with {df_requests.shape[0]} context vectors")
                for row_idx, row in df_requests.iterrows():
                    start_timestamp, end_timestamp, ctx_vector = row['start_timestamp'], row['end_timestamp'], row[
                        'ctx_vector']
                    try:
                        row_context_response = context_predictor.predict(ctx_vector)
                        context_responses_id.append([start_timestamp, end_timestamp, row_context_response])
                    except KeyboardInterrupt:
                        sys.exit(1)
                    except:
                        logger.info(f"Error for {int(start_timestamp)}, {int(end_timestamp)}, {str(ctx_vector)}")
                context_responses_dict[id] = pd.DataFrame(context_responses_id,
                                                          columns=['start_timestamp', 'end_timestamp', 'context'])
                # logger.info("Empty context cache to process new user")
                # context_predictor.empty_buffer()

        time_str = datetime.now().strftime("%Y%m%d")
        context_responses_filepath = f"{run_config.cache_dir}/{run_config.experiment}/context_responses_{time_str}.pb"
        pickle.dump(context_responses_dict, open(context_responses_filepath, 'wb'))

    elif run_mode == 'testing':
        ...
        # dummy testing context prediction
        # st_times = int(time.time())
        # timestamps = np.arange(st_times, st_times + (800*50), 50)
        #
        # # for i in range(40):
        # #     timestamps.append(int(time.time()))
        # #     time.sleep(1)
        # activity_list = ["Lying down", "Sitting", "Walking", "Running", "Bicycling", "Sleeping", "Lab work", "In class",
        #                  "In a meeting", "Drive - I'm the driver", "Drive - I'm a passenger", "Exercise", "Cooking",
        #                  "Shopping", "Strolling", "Drinking (alcohol)", "Bathing - shower", "Cleaning", "Doing laundry",
        #                  "Washing dishes", "WatchingTV", "Surfing the internet", "Singing", "Talking", "Computer work",
        #                  "Eating", "Toilet", "Grooming", "Dressing", "Stairs - going up", "Stairs - going down", "Standing",
        #                  "With co-workers", "With friends"]
        # activities = []
        # for i in range(400):
        #     num_pars = np.random.randint(low=1, high=4)
        #     activities_arr = np.random.choice(activity_list, num_pars)
        #     activities.append(','.join(activities_arr))
        #
        # st = time.time()
        # responses = []
        # for i in range(400):
        #     responses.append(process_request(timestamps[i], activities[i]))
