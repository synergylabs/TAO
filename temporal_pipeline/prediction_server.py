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
from init_server_components import init_extrasensory_predictor, init_casas_predictor, init_wellness_predictor, initialize_logger

# from test_configs.testing_sep182023 import global_config
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/extrasensory', methods=['GET'])
def process_extrasensory_request():
    ts = request.args.get('timestamp')
    activities = request.args.get('activities')
    request_dict = {
        'timestamp': int(ts),
        'activities': activities.split(',')
    }
    context_label, activity_inputs = extra_context_predictor.process_request(request_dict)

    response_dict = {'timestamp': ts,
                     'context': context_label,
                     'activity_input': activity_inputs,
                     'dataset': 'extrasensory'}
    return response_dict


@app.route('/casas', methods=['GET'])
def process_casas_request():
    ts = request.args.get('timestamp')
    activities = request.args.get('activities')
    request_dict = {
        'timestamp': int(ts),
        'activities': activities.split(',')
    }
    context_label, activity_inputs = casas_context_predictor.process_request(request_dict)

    response_dict = {'timestamp': ts,
                     'context': context_label,
                     'activity_input': activity_inputs,
                     'dataset': 'casas'}
    return response_dict

@app.route('/wellness/stress',methods=['GET'])
def process_stress_request():
    ts = request.args.get('timestamp')
    contexts = request.args.get('contexts')
    request_dict = {
        'timestamp': int(ts),
        'contexts': contexts.split(',')
    }
    stress_score, context_inputs = wellness_predictor.process_request(request_dict)

    response_dict = {'timestamp': ts,
                     'stress_score': stress_score,
                     'context_input': context_inputs,
                     'method': 'fsm'}
    return response_dict


if __name__ == '__main__':
    # Initialize the logger
    log_dir = 'cache/logs/prediction'
    logger = initialize_logger(log_dir)

    # load global config
    global_config = json.load(open('prediction_config.json', 'r'))
    cluster_counts = {
        'casas': {
            '5': 17,
            '10': 15,
            '30': 12,
            '60': 15,
        },
        'extrasensory': {
            '5': 25,
            '10': 29,
            '30': 13,
            '60': 15,
        }
    }

    input_sizes = {
        'casas': {
            '5': 105,
            '10': 210,
            '30': 210,
            '60': 210,
        },
        'extrasensory': {
            '5': 145,
            '10': 290,
            '30': 290,
            '60': 290,
        }
    }

    # Initialize dataset predictors
    extra_context_predictor, extra_run_config = init_extrasensory_predictor(global_config, cluster_counts, input_sizes,
                                                                            logger)
    casas_context_predictor, casas_run_config = init_casas_predictor(global_config, cluster_counts, input_sizes, logger)

    wellness_predictor, wellness_run_config = init_wellness_predictor(global_config, logger)

    app.run('0.0.0.0', port=8080, debug=True)
