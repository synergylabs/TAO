'''
This is initialization of context recognition module.
This file have wrapper functions to train and evalute temporal context recognition.
'''

# core library functions
from datetime import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import confusion_matrix

# custom libraries
from .dataloaders import load_data_parser

# representation learning
from .architectures import fetch_re_model
from .representationTrainer import representationTrainer

# Clustering
from .clustering import fetch_cluster_model
from .clusterTrainer import clusterTrainer

from .labelling import fetch_labeler, directLabeler
from .contextLabeler import contextLabeler

from utils import time_diff


def fetch_input_data(run_config, logger):
    """
    Get input data for training post processing raw activity data
    :param run_config:
    :param logger:
    :return: np.ndarray with data loaded
    """

    data_parse_func = load_data_parser(run_config.dataset, run_config.parse_style, logger)

    t_dataparse_start = datetime.now()

    X, X_deduped, context_request_dict = data_parse_func(rawdatasrc=run_config.datafile,
                                                         args=run_config,
                                                         cache_dir=run_config.cache_dir,
                                                         access_cache=run_config.access_cache,
                                                         logger=logger)

    t_dataparse_end = datetime.now()

    logger.info(f"Got parsed data for training in {time_diff(t_dataparse_start, t_dataparse_end)} secs.")

    return X, X_deduped, context_request_dict


def fetch_prediction_requests(run_config, prediction_parser, logger):
    '''
    Get input requests for prediction service from datasets
    :param run_config:
    :param logger:
    :return:
    '''
    data_parse_func = load_data_parser(run_config.dataset, prediction_parser, logger)

    t_dataparse_start = datetime.now()

    context_requests_dict = data_parse_func(rawdatasrc=run_config.datafile,
                                            args=run_config,
                                            cache_dir=run_config.cache_dir,
                                            access_cache=run_config.access_cache,
                                            logger=logger)

    t_dataparse_end = datetime.now()

    logger.info(f"Got parsed requests for prediction in {time_diff(t_dataparse_start, t_dataparse_end)} secs.")

    return context_requests_dict


def learn_context_representations(run_config, data_raw, logger):
    """
    Learn representations from context on raw data
    :param run_config: main run config
    :param data_raw: raw data to learn embedding over
    :param logger: logging object
    :return: representation learning models, data representations
    """
    t_ae_learning_start = datetime.now()

    # fetch model architecture
    model_re_arch = fetch_re_model(model_name=run_config.model_re,
                                   logger=logger)

    # initialize model and load dataset
    model_re = representationTrainer(model_arch=model_re_arch,
                                     input_size=data_raw.shape[1],
                                     embedding_size=run_config.embedding_size,
                                     run_config=run_config,
                                     logger=logger)

    if hasattr(model_re, "n_hidden"):
        run_config.embedding_size = model_re.n_hidden

    training_metrics = []
    prev_epoch_accuracy = 0.
    training_relative_tol = 0.5
    max_acc_threshold = 80
    max_stale_epochs = 10
    stale_epochs = 0.
    for epoch in range(1, run_config.num_epochs_ae + 1):
        model_re.load_dataset(data_raw)
        epoch_train_loss, epoch_train_accuracy = model_re.train(epoch)
        epoch_test_loss, epoch_test_accuracy = model_re.test(epoch)
        if (epoch_test_accuracy > max_acc_threshold) & (
                (epoch_test_accuracy - prev_epoch_accuracy) < training_relative_tol):
            stale_epochs += 1
        else:
            stale_epochs = 0.
        if stale_epochs >= max_stale_epochs:
            logger.info(
                f"Early Stopping criteria met by more than {max_stale_epochs} stale epochs, exiting training...")
            break
        prev_epoch_accuracy = epoch_test_accuracy
        training_metrics.append({
            'epoch': epoch,
            'batch_size': model_re.train_loader.batch_size,
            'train_samples': len(model_re.train_loader.dataset),
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_accuracy,
            'test_samples': len(model_re.test_loader.dataset),
            'test_loss': epoch_test_loss,
            'test_acc': epoch_test_accuracy
        })
    logger.warning(str(training_metrics[-1]))

    model_re.save()
    data_embeddings = model_re.get_embedding(data_raw)

    t_ae_learning_end = datetime.now()
    logger.info(f"Got data embeddings for clustering in {time_diff(t_ae_learning_start, t_ae_learning_end)} secs.")

    return model_re, data_embeddings, training_metrics


def run_context_clustering(run_config, context_object, logger):
    """
    Cluster embeddings to bucket them into variety of contexts
    :param run_config: main run config
    :param context_object: raw data and embeddings to learn context clusters
    :param logger: logging object
    :return: clustering model, cluster labels for input data
    """
    model_cluster, data_labels, cluster_representations = None, None, None

    t_clustering_start = datetime.now()

    model_cluster_arch = fetch_cluster_model(run_config.model_cluster, logger)

    # initialize clustering model

    model_cluster = clusterTrainer(model_cluster=model_cluster_arch,
                                   input_size=context_object['data_raw'].shape[1],
                                   embedding_size=run_config.embedding_size,
                                   run_config=run_config,
                                   logger=logger)

    # fit clustering model based on rawdata or embeddings
    if model_cluster.is_input_raw:
        clustering_in_data = context_object['data_raw']
    else:
        clustering_in_data = context_object['data_embeddings']

    data_labels = model_cluster.fit_predict(clustering_in_data)
    cluster_representations = model_cluster.get_context_representations()

    t_clustering_end = datetime.now()

    logger.info(f"Got context clusters without labels in {time_diff(t_clustering_start, t_clustering_end)} secs.")

    return model_cluster, data_labels, cluster_representations


def label_context_clusters(run_config, context_object, logger):
    """
    Label context clusters based on ontological information provided from third party
    :param run_config: main run config
    :param context_object: clustering model and ontological information to learn clusters labels
    :param logger: logging object
    :return: context_labels for different clusters
    """

    context_labels = None

    t_labeling_start = datetime.now()

    # todo: Code for context labelling
    model_re = context_object['model_re']

    # fetch accurate labeling model
    labeler_model = fetch_labeler(labeler_name=run_config.model_labeler, logger=logger)
    direct_labeler_model = fetch_labeler(labeler_name='onto_conv', logger=logger)

    # initialize context labeller

    context_labeler = contextLabeler(labeler_model=labeler_model,
                                     run_config=run_config,
                                     logger=logger)
    direct_labeler = contextLabeler(labeler_model=direct_labeler_model,
                                    run_config=run_config,
                                    logger=logger)

    cluster_representations = context_object['cluster_representations']
    decoded_cluster_representations = model_re.get_reconstructed_input(cluster_representations)
    direct_labeler.label_clusters(decoded_cluster_representations)
    decoded_centroid_direct_labels = direct_labeler.get_all_labels()

    context_representations = np.zeros_like(decoded_cluster_representations)
    context_positional_counts = np.zeros_like(decoded_cluster_representations)

    X = context_object['data_raw']
    data_labels = context_object['data_labels']
    for cluster_id in range(context_representations.shape[0]):
        positional_counts_cluster = np.sum(X[data_labels == cluster_id], axis=0)
        X_cluster = np.mean(X[data_labels == cluster_id], axis=0)
        perc_98 = np.percentile(X_cluster, 98)
        X_cluster[X_cluster > perc_98] = 1
        X_cluster[X_cluster <= perc_98] = 0
        # X_cluster = np.percentile(X[data_labels == cluster_id],75,axis=0)
        context_representations[cluster_id, :] = deepcopy(X_cluster)
        context_positional_counts[cluster_id, :] = deepcopy(positional_counts_cluster)

    context_labeler.label_clusters(context_representations)
    context_onto_labels = context_labeler.get_all_labels()

    direct_labeler.label_clusters(context_representations)
    context_direct_labels = direct_labeler.get_all_labels()

    t_labeling_end = datetime.now()

    context_labeling_info = {
        'context_representations': context_representations.tolist(),
        'context_positional_counts': context_positional_counts.tolist(),
        'direct_labels': context_direct_labels,
        'onto_labels': context_onto_labels,
        'decoded_representations': decoded_cluster_representations.tolist(),
        'decoded_labels': decoded_centroid_direct_labels
    }

    logger.info(f"Got context labels for clusters in {time_diff(t_labeling_start, t_labeling_end)} secs.")

    return context_labeler, context_labeling_info


def get_predictions(run_config, context_object, logger):
    """
    Get prediction from context requests stored by converting it into timestamped information
    :param run_config:
    :param context_object:
    :param logger:
    :return: dataframe with timestamped cluster_id information
    """
    logger.info("Starting timestamped level predictions...")
    t_predict_start = datetime.now()
    model_re = context_object['model_re']
    model_cluster = context_object['model_cluster']

    # Get instance level predictions
    X = context_object['data_raw']
    data_labels = context_object['data_labels']
    df_instance_predictions = (X.tolist(), data_labels.tolist())
    logger.info("Compiled unique instance level prediction")
    context_requests = context_object['context_request_dict']
    timestamped_predictions = dict()
    for key_id in context_requests.keys():
        df_ctx_response_id = context_requests[key_id]
        if df_ctx_response_id.shape[0] > 0:
            if model_cluster.is_input_raw:
                cluster_ids = model_cluster.predict(np.stack(df_ctx_response_id['ctx_vector'].values))
            else:
                ctx_embeddings = model_re.get_embedding(np.stack(df_ctx_response_id['ctx_vector'].values))
                cluster_ids = model_cluster.predict(ctx_embeddings)
            df_ctx_response_id['cluster_id'] = cluster_ids
            df_ctx_response_id = df_ctx_response_id.sort_values(by='start_timestamp').reset_index(drop=True)
            next_start_timestamp = np.zeros(df_ctx_response_id.shape[0])
            next_start_timestamp[:-1] = df_ctx_response_id.iloc[1:]['start_timestamp']
            next_start_timestamp[-1] = df_ctx_response_id.iloc[-1]['end_timestamp']
            df_ctx_response_id['next_start_timestamp'] = next_start_timestamp
            if 'isTrain' in df_ctx_response_id.columns:
                df_ctx_response_id = df_ctx_response_id[
                    ['isTrain', 'start_timestamp', 'next_start_timestamp', 'end_timestamp', 'cluster_id']]
            else:
                df_ctx_response_id = df_ctx_response_id[
                    ['start_timestamp', 'next_start_timestamp', 'end_timestamp', 'cluster_id']]
            timestamped_predictions[key_id] = df_ctx_response_id

            logger.info(f"Got timestamped prediction for id: {key_id}")

    t_predict_end = datetime.now()
    logger.info(f"Got timestamped predictions in {time_diff(t_predict_start, t_predict_end)} secs.")
    return df_instance_predictions, timestamped_predictions


def predict_cluster_id(model_re, model_cluster, ctx_vector):
    '''
    Predict cluster_id for given representation and clustering model
    :param model_re:
    :param model_cluster:
    :param ctx_vector:
    :return:
    '''
    if model_cluster.is_input_raw:
        ctx_cluster_id = model_cluster.predict(ctx_vector.reshape(1, -1))[0]
    else:
        ctx_embedding = model_re.get_embedding(ctx_vector)
        ctx_cluster_id = model_cluster.predict(ctx_embedding.reshape(1, -1))[0]
    return ctx_cluster_id


def run_ontological_evaluation(run_config, context_object, logger):
    """
    Todo:FUNCTION OBSELETE. Need to remove.
    Get training accuracy for each sample considering ontological contexts
    as ground truth.
    :param run_config: main run config
    :param context_object: context_labeller and
    :param logger: logging object
    :return: training accuracy results
    """
    training_accuracy_results = None
    # get relevant objects for getting predicted contexts.
    context_labeler = context_object['context_labeler']
    ontolist = context_labeler.get_ontolist()
    X = context_object['data_raw']
    pred_cluster_labels = context_object['data_labels']
    pred_context = [context_labeler.get_cluster_label(xr) for xr in pred_cluster_labels]
    onto_context = []
    for i in range(X.shape[0]):
        x_row = X[i, :]

        x_activities = []
        activity_labels = np.array(run_config.activity_labels)
        num_activities = len(run_config.activity_labels)

        for activity_idx, activity_vec in enumerate(np.reshape(x_row, (-1, num_activities))):
            activity_set = activity_labels[np.where(activity_vec)[0]].tolist()
            activity_set_ontolist = []
            if len(activity_set) > 0:
                # convert activities to ontological activities
                for activity in activity_set:
                    activity_ontolist = ontolist.activity_mapping[run_config.dataset][activity]
                    activity_set_ontolist.append(activity_ontolist)
                x_activities.append(activity_set_ontolist)
            else:
                x_activities.append(['none'])
        if len(x_activities) > 0:
            x_onto_context = context_labeler.labeler.get_onto_label(x_activities)
        else:
            x_onto_context = f'None:1.0'
        onto_context.append(x_onto_context)

    tight_accuracy_score = []  # if 1 if all the contexts are similar
    spot_accuracy_score = []  # is 1 if any of the onto context is available
    maj_accuracy_score = []  # 1 if most probable context in pred and onto are same
    kl_divergence_score = []  # KL divergence between score confidences
    pred_maj_ctx, onto_maj_ctx = [], []
    for idx, (y_pred, y_onto) in enumerate(zip(pred_context, onto_context)):
        y_pred = y_pred.split("__")[-1]  # remove leading cluster id
        pred_ctx_scores = {key: 0. for key in ontolist.context_mapping.keys()}
        pred_ctx = []
        for context_category in y_pred.split("_"):
            context_name, context_score = context_category.split(":")
            if (not (context_name == 'None')) and (context_name not in pred_ctx):
                pred_ctx_scores[context_name] += float(context_score)
                pred_ctx.append(context_name)

        onto_ctx_scores = {key: 0. for key in ontolist.context_mapping.keys()}
        onto_ctx = []
        for context_category in y_onto.split("_"):
            context_name, context_score = context_category.split(":")
            if (not (context_name == 'None')) and (context_name not in onto_ctx):
                onto_ctx_scores[context_name] += float(context_score)
                onto_ctx.append(context_name)

        # append scores based on context scores.

        epsilon = 1e-2
        kl_divergence = 0.
        spot_accuracy = 0.
        spot_count = 0.
        for context_name in onto_ctx_scores.keys():
            if (not (context_name == 'None')) and (onto_ctx_scores[context_name] > 0.):
                kl_divergence += onto_ctx_scores[context_name] * \
                                 np.log(onto_ctx_scores[context_name] / (pred_ctx_scores[context_name] + epsilon))
                spot_accuracy += (context_name in pred_ctx)
                spot_count += 1

        tight_accuracy_score.append(sorted(pred_ctx) == sorted(onto_ctx))
        pred_maj_score_context_name = max(pred_ctx_scores, key=pred_ctx_scores.get)
        pred_maj_ctx.append(pred_maj_score_context_name)
        onto_maj_score_context_name = max(onto_ctx_scores, key=onto_ctx_scores.get)
        onto_maj_ctx.append(onto_maj_score_context_name)
        if (not (pred_maj_score_context_name == 'None')):
            maj_accuracy_score.append(pred_maj_score_context_name == onto_maj_score_context_name)
        else:
            maj_accuracy_score.append(0.)

        if spot_count > 0.:
            spot_accuracy_score.append(spot_accuracy / spot_count)
        else:
            spot_accuracy_score.append(0.)
        kl_divergence_score.append(kl_divergence)

    ctx_labels = list(ontolist.context_mapping.keys())
    maj_ctx_confusion_matrix = confusion_matrix(onto_maj_ctx, pred_maj_ctx, labels=ctx_labels)

    accuracy_results_overall = {
        'onto_predicted_contexts': pred_context,
        'onto_context': onto_context,
        'onto_tight_scores': tight_accuracy_score,
        'onto_spot_scores': spot_accuracy_score,
        'onto_maj_scores': maj_accuracy_score,
        'onto_kl_divergence': kl_divergence_score,
        'onto_pred_maj_ctx': pred_maj_ctx,
        'onto_maj_ctx': onto_maj_ctx
    }

    accuracy_results_summary = {
        'onto_overall_tight_accuracy': sum(tight_accuracy_score) * 100 / len(tight_accuracy_score),
        'onto_overall_spot_score': sum(spot_accuracy_score) * 100 / len(spot_accuracy_score),
        'onto_overall_maj_score': sum(maj_accuracy_score) * 100 / len(maj_accuracy_score),
        'onto_average_kl_divergence': sum(kl_divergence_score) / len(kl_divergence_score),
        'onto_maj_ctx_confusion_matrix': maj_ctx_confusion_matrix.tolist(),
        'onto_labels_confusion_matrix': ctx_labels
    }

    return accuracy_results_overall, accuracy_results_summary


def run_gt_evaluation(run_config, context_object, logger):
    """
    Get training accuracy for each sample with respect to manual contexts
    as ground truth.
    :param run_config: main run config
    :param context_object: context_labeller and
    :param logger: logging object
    :return: training accuracy results
    """
    training_accuracy_results = None
    # get relevant objects for getting predicted contexts.
    context_labeler = context_object['context_labeler']
    ontolist = context_labeler.get_ontolist()
    model_re = context_object['model_re']
    model_cluster = context_object['model_cluster']

    X_labelled = context_object['data_labelled']
    pred_context = []
    gt_context = []
    for ctx_vector, ctx_gt_arr in X_labelled:
        if model_cluster.is_input_raw:
            ctx_cluster_id = model_cluster.predict(ctx_vector.reshape(1, -1))[0]
        else:
            ctx_embedding = model_re.get_embedding(ctx_vector)
            ctx_cluster_id = model_cluster.predict(ctx_embedding.reshape(1, -1))[0]
        ctx_label = context_labeler.get_cluster_label(ctx_cluster_id)
        pred_context.append(ctx_label)
        for i in range(len(ctx_gt_arr)):
            # print('|'+ctx_gt_arr[i][2]+'|')
            if ctx_gt_arr[i][2] == 'Meal_Preparation':
                ctx_gt_arr[i][2] == 'PreparingMeal'
                # print(ctx_gt_arr[i][2])
        ctx_label_gt = "_".join([f"{xr[2]}:{1 / len(ctx_gt_arr)}" for xr in ctx_gt_arr])
        gt_context.append(ctx_label_gt)

    tight_accuracy_score = []  # if 1 if all the contexts are similar
    spot_accuracy_score = []  # is 1 if any of the onto context is available
    maj_accuracy_score = []  # 1 if most probable context in pred and onto are same
    kl_divergence_score = []  # KL divergence between score confidences
    pred_maj_ctx, gt_maj_ctx = [], []
    for idx, (y_pred, y_gt) in enumerate(zip(pred_context, gt_context)):
        y_pred = y_pred.split("__")[-1]  # remove leading cluster id
        pred_ctx_scores = {key: 0. for key in ontolist.context_mapping.keys()}
        pred_ctx = []
        for context_category in y_pred.split("_"):
            context_name, context_score = context_category.split(":")
            if (not (context_name == 'None')) and (context_name not in pred_ctx):
                pred_ctx_scores[context_name] += float(context_score)
                pred_ctx.append(context_name)

        gt_ctx_scores = {key: 0. for key in ontolist.context_mapping.keys()}
        gt_ctx = []
        for context_category in y_gt.split("_"):
            try:
                context_name, context_score = context_category.split(":")
            except:
                print(y_gt)
                context_name, context_score = 'None', '1.0'
            if (not (context_name == 'None')) and (context_name not in gt_ctx):
                gt_ctx_scores[context_name] += float(context_score)
                gt_ctx.append(context_name)

        # append scores based on context scores.

        epsilon = 1e-2
        kl_divergence = 0.
        spot_accuracy = 0.
        spot_count = 0.
        for context_name in gt_ctx_scores.keys():
            if (not (context_name == 'None')) and (gt_ctx_scores[context_name] > 0.):
                kl_divergence += gt_ctx_scores[context_name] * \
                                 np.log(gt_ctx_scores[context_name] / (pred_ctx_scores[context_name] + epsilon))
                spot_accuracy += (context_name in pred_ctx)
                spot_count += 1

        tight_accuracy_score.append(sorted(pred_ctx) == sorted(gt_ctx))
        pred_maj_score_context_name = max(pred_ctx_scores, key=pred_ctx_scores.get)
        pred_maj_ctx.append(pred_maj_score_context_name)
        gt_maj_score_context_name = max(gt_ctx_scores, key=gt_ctx_scores.get)
        gt_maj_ctx.append(gt_maj_score_context_name)
        if (not (pred_maj_score_context_name == 'None')):
            maj_accuracy_score.append(pred_maj_score_context_name == gt_maj_score_context_name)
        else:
            maj_accuracy_score.append(0.)

        if spot_count > 0.:
            spot_accuracy_score.append(int(spot_accuracy > 0))
        else:
            spot_accuracy_score.append(0.)
        kl_divergence_score.append(kl_divergence)

    ctx_labels = list(ontolist.context_mapping.keys())
    maj_ctx_confusion_matrix = confusion_matrix(gt_maj_ctx, pred_maj_ctx, labels=ctx_labels)

    accuracy_results_overall = {
        'gt_predicted_contexts': pred_context,
        'gt_context': gt_context,
        'gt_tight_scores': tight_accuracy_score,
        'gt_spot_scores': spot_accuracy_score,
        'gt_maj_scores': maj_accuracy_score,
        'gt_kl_divergence': kl_divergence_score,
        'gt_pred_maj_ctx': pred_maj_ctx,
        'gt_maj_ctx': gt_maj_ctx
    }

    accuracy_results_summary = {
        'gt_overall_tight_accuracy': sum(tight_accuracy_score) * 100 / len(tight_accuracy_score),
        'gt_overall_spot_score': sum(spot_accuracy_score) * 100 / len(spot_accuracy_score),
        'gt_overall_maj_score': sum(maj_accuracy_score) * 100 / len(maj_accuracy_score),
        'gt_average_kl_divergence': sum(kl_divergence_score) / len(kl_divergence_score),
        'gt_maj_ctx_confusion_matrix': maj_ctx_confusion_matrix.tolist(),
        'gt_labels_confusion_matrix': ctx_labels
    }

    return accuracy_results_overall, accuracy_results_summary
