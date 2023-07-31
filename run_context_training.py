'''
This is main file to run cntext training based on given config
Author: Prasoon Patidar
Created At: Apr 28, 2022
'''

# core libraries
import sys
import os
import logging
import traceback
from datetime import datetime
from logging.handlers import WatchedFileHandler
import pickle
import json
from collections import namedtuple
import numpy as np

# custom libraries
import pandas as pd

from utils import time_diff, get_config_from_json
import context_recognition
from wellness.stress_predictors.fsm_stress_pipeline import get_stress_score


def run_context_training(config, base_config, logger):
    '''
    This is main function to run context training based on provided configurations. Default configurations are provided
    in default config file
    :param config: Main config file provided by developer
    :param base_config: Default file with complete set of config requirements
    :param logger: Default logger object

    :return: None
    '''
    logger.critical(f"-------------------------Initiating Experiment: {config['experiment']}-------------------------")
    # Get config dict from default and config parameters

    run_config_dict = get_config_from_json(base_config)
    run_config_dict.update(config)
    run_config = namedtuple('run_config', run_config_dict.keys())(*run_config_dict.values())

    if not os.path.exists(run_config.cache_dir):
        os.makedirs(run_config.cache_dir)

    # ============== Start Context Training Pipeline ==============

    context_object = dict()
    logger.info("Collecting and preprocessing data...")

    # -------------- Input data fetching and preprocessing --------------
    X, X_deduped, context_request_dict = context_recognition.fetch_input_data(run_config, logger)
    logger.info(f"Dataset dimensions: {X.shape} ")

    if run_config.parse_unique_data:
        logger.info("using De-duplicated parsed data...")
        context_object["data_raw"] = X_deduped
    else:
        context_object["data_raw"] = X
    logger.info(f"Dataset dimensions: {context_object['data_raw'].shape}")
    # Some RE Networks needs to know input dimensions and datatypes
    run_config_dict.update({'data_sample': X[0, :].tolist()})
    #if we use a subset on labels, update ontology labels in config
    if 'onto_activity_labels' in context_request_dict.keys():
        run_config_dict.update({'onto_activity_labels': context_request_dict['onto_activity_labels']})
        del context_request_dict['onto_activity_labels']
    context_object["context_request_dict"] = context_request_dict
    run_config = namedtuple('run_config', run_config_dict.keys())(*run_config_dict.values())

    # -------------- Representation learning --------------

    model_re, data_embeddings, repr_training_metrics = context_recognition.learn_context_representations(run_config,
                                                                                                         context_object[
                                                                                                             "data_raw"],
                                                                                                         logger)
    model_re.save()
    context_object['model_re'] = model_re
    context_object['data_embeddings'] = data_embeddings
    context_object['repr_training_metrics'] = repr_training_metrics
    repr_info_dict = {
        'repr_training_metrics': repr_training_metrics,
        'data_embeddings': data_embeddings.tolist()
    }

    # -------------- Context Clustering --------------

    model_cluster, data_labels, cluster_representations = context_recognition.run_context_clustering(run_config,
                                                                                                     context_object,
                                                                                                     logger)
    model_cluster.save()
    context_object['model_cluster'] = model_cluster
    context_object['data_labels'] = data_labels
    context_object['cluster_representations'] = cluster_representations
    clustering_info_dict = {
        'cluster_representations': cluster_representations.tolist(),
    }

    # -------------- Context Timestamped Prediction --------------

    instance_predictions, timestamped_predictions = context_recognition.get_predictions(run_config, context_object,
                                                                                        logger)
    prediction_info_dict = {
        'instance_predictions': instance_predictions,
    }

    # -------------- Context Labelling --------------

    context_labeler, labeling_info_dict = context_recognition.label_context_clusters(run_config, context_object, logger)
    context_labeler.save()
    context_object.update(labeling_info_dict)

    # -------------- Training Evaluation --------------

    eval_info_dict = dict()
    if run_config.evaluate_gtlabels:
        gt_accuracy_overall, gt_accuracy_summary = context_recognition.run_gt_evaluation(run_config,
                                                                                         context_object,
                                                                                         logger)
        eval_info_dict.update(gt_accuracy_summary)

    # -------------- Caching training results --------------

    results_summary = dict()
    results_summary.update({
        'run_config': dict(run_config._asdict()),
    })
    results_summary.update(repr_info_dict)
    results_summary.update(clustering_info_dict)
    results_summary.update(prediction_info_dict)
    results_summary.update(labeling_info_dict)

    if run_config.access_cache | (run_config.parse_style == 'incremental') | (run_config.dataset == 'realworld'):
        exp_cache_dir = f'{run_config.cache_dir}/{run_config.experiment}/results'
        if not os.path.exists(exp_cache_dir):
            os.makedirs(exp_cache_dir)
        file_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        json.dump(results_summary, open(f'{exp_cache_dir}/results_summary_{file_suffix}.json', 'w'))
        pickle.dump(timestamped_predictions, open(f'{exp_cache_dir}/timestamped_predictions_{file_suffix}.pb', 'wb'))

    # Collect wellness results if allowed
    if run_config.evaluate_wellness:
        # get parsed data for wellness
        wellness_data_parse_func = context_recognition.load_data_parser(run_config.dataset, 'wellness', logger)
        df_parsed_wellness_data = wellness_data_parse_func(rawdatasrc=run_config.datafile,
                                                           args=run_config,
                                                           cache_dir=run_config.cache_dir,
                                                           access_cache=run_config.access_cache,
                                                           logger=logger)

        weekly_stress_scores = None
        # get context for parsed wellness data
        # Todo: Current version does not worry about missing data
        for id in df_parsed_wellness_data['id'].unique():
            df_parsed_wellness_data_id = df_parsed_wellness_data[df_parsed_wellness_data.id == id]
            time_context_list = []
            for row_idx in range(0, df_parsed_wellness_data_id.shape[0], run_config.sliding_parameter):
                df_window = df_parsed_wellness_data_id.iloc[row_idx:row_idx + run_config.lag_parameter]
                ctx_input_vector = np.expand_dims(np.concatenate(df_window['activity_vec'].values), axis=0)
                if ctx_input_vector.shape[1] == model_re.model.input_size:
                    if model_cluster.is_input_raw:
                        clustering_in_data = ctx_input_vector
                    else:
                        clustering_in_data = model_re.get_embedding(ctx_input_vector)
                    ctx_cluster_id = model_cluster.predict(np.expand_dims(clustering_in_data, axis=0))
                    ctx_cluster_label = context_labeler.get_cluster_label(int(ctx_cluster_id))
                    ctx_cluster_label = ctx_cluster_label.split("__")[-1]  # remove leading cluster id

                    pred_ctx_names = [k.split(":")[0] for k in ctx_cluster_label.split("_") if
                                      (not (k.split(":")[0] == 'None'))]
                    pred_ctx_scores = [float(k.split(":")[1]) for k in ctx_cluster_label.split("_") if
                                       not (k.split(":")[0] == 'None')]
                    ctx_name = 'None'
                    if len(pred_ctx_names) > 0:
                        ctx_name = pred_ctx_names[np.argmax(pred_ctx_scores)]
                    # get majority context_name based on label
                    for ts in df_window.timestamp.values:
                        time_context_list.append([ts, ctx_name])
                else:
                    for ts in df_window.timestamp.values:
                        time_context_list.append([ts, np.nan])
            df_user_wellness_input = pd.DataFrame(time_context_list, columns=['timestamp', 'context'])
            df_user_wellness_input['context'] = df_user_wellness_input.context.fillna(method='ffill')
            weekly_stress_score_id = get_stress_score(df_user_wellness_input, run_config, logger)
            weekly_stress_score_id['id'] = id
            weekly_stress_score_id = weekly_stress_score_id[['id', 'week', 'stress_score']]
            if weekly_stress_scores is None:
                weekly_stress_scores = weekly_stress_score_id.copy(deep=True)
            else:
                weekly_stress_scores = pd.concat([weekly_stress_scores, weekly_stress_score_id], ignore_index=True)

        context_object['weekly_stress_scores'] = weekly_stress_scores
    logger.critical(f"-------------------------Finished Context Training: {config['experiment']}-------------------------")
    # del logger, logger_master, context_object
    return None


if __name__ == '__main__':
    # from train_configs.sep10_config import get_configs
    from train_configs.sep10_inc_config import get_configs
    # from train_configs.sep12_realworld_config import get_configs
    # from train_configs.sep13_mites_config import get_configs

    final_configs = get_configs()

    # Initialize the logger

    logger_master = logging.getLogger('context_training')
    logger_master.setLevel(logging.INFO)
    log_dir = final_configs[0]['log_dir']

    # Add core logger handler
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(module)s | %(funcName)s:L%(lineno)d | %(levelname)s | %(message)s')
    date_str = datetime.now().strftime("%Y_%M_%d")
    core_logging_handler = WatchedFileHandler(f"{log_dir}/context_training_{date_str}.log")
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    # Add stdout logger handler
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(module)s | %(funcName)s:L%(lineno)d | %(levelname)s | %(message)s')
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(console_formatter)
    logger_master.addHandler(console_log)
    rlogger = logging.LoggerAdapter(logger_master, {})

    for rconfig in final_configs:
        rbase_config = rconfig['base_config']
        # run_context_training(rconfig, rbase_config, rlogger)
        try:
            run_context_training(rconfig, rbase_config, rlogger)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            rlogger.info(f"Error in running experiment {rconfig['experiment']}, {str(traceback.format_exc())}")
            with open(f'{log_dir}/error_experiments.log', 'a+') as f:
                f.write(f"{rconfig['experiment']},{str(traceback.format_exc())}\n")
