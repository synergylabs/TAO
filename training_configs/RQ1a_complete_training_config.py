from copy import deepcopy

data_configs = [
    ## Extrasensory
    {'dataset': 'extrasensory', 'base_config': 'context_configs/extrasensory.json',
     'lag_parameter': 5, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 135},
    {'dataset': 'extrasensory', 'base_config': 'context_configs/extrasensory.json',
     'lag_parameter': 10, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 270},
    # {'dataset': 'extrasensory', 'base_config': 'context_configs/extrasensory.json',
    #  'lag_parameter': 30, 'merge_mins': 3, 'max_time_interval_in_mins': 60, 'input_size': 270},
    # {'dataset': 'extrasensory', 'base_config': 'context_configs/extrasensory.json',
    #  'lag_parameter': 60, 'merge_mins': 6, 'max_time_interval_in_mins': 180, 'input_size': 270},
    #
    # # # # Casas
    # {'dataset': 'casas', 'base_config': 'context_configs/casas.json',
    #  'lag_parameter': 5, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 105},
    # {'dataset': 'casas', 'base_config': 'context_configs/casas.json',
    #  'lag_parameter': 10, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 210},
    # {'dataset': 'casas', 'base_config': 'context_configs/casas.json',
    #  'lag_parameter': 30, 'merge_mins': 3, 'max_time_interval_in_mins': 60, 'input_size': 210},
    # {'dataset': 'casas', 'base_config': 'context_configs/casas.json',
    #  'lag_parameter': 60, 'merge_mins': 6, 'max_time_interval_in_mins': 180, 'input_size': 210},

    #Real World
    # {'dataset': 'realworld', 'base_config': 'context_configs/realworld.json',
    #  'lag_parameter': 0.5, 'merge_mins': 0.05, 'max_time_interval_in_mins': 1}

]

model_configs = [
    {'model_re': 'TAE', 'stacked_input': False},
]

global_config = {
    'device': 'cuda:0',
    "parse_style": "combined",
    'sliding_parameter': 1,
    "cache_dir": "cache/rq1a",
    "log_dir": "cache/logs/rq1a",
    'model_cluster': 'DCN',
    'num_epochs_ae': 5,
    'num_epochs_cluster': 100,
    'reconstruction_conf_val': 0.5,
    'lr_ae': 1e-2,
    'lr_cluster': 1e-3,
    'cnet_type': 'kmeans',
    "fcn_hidden_size": 256,
    'cnet_n_clusters': 30,
    'cnet_cluster_loss_ratio': 0.8,
    'kmeans_min_clusters': 10,
    'kmeans_max_clusters': 40,
    'model_labeler': 'onto_conv',
}


def get_configs():
    final_configs = []
    for dconfig in data_configs:
        for mconfig in model_configs:
            new_config = deepcopy(global_config)
            new_config.update(dconfig)
            new_config.update(mconfig)
            experiment_name = f"{dconfig['dataset']}_{dconfig['lag_parameter']}_"
            experiment_name += f"{dconfig['merge_mins']}_{mconfig['model_re']}"
            new_config['experiment'] = experiment_name
            final_configs.append(new_config)
            print(new_config)
    return final_configs
