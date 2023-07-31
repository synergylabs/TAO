from copy import deepcopy
import glob

extrasensory_base_config = {'dataset': 'extrasensory', 'base_config': 'context_configs/extrasensory.json',
                            'lag_parameter': 5, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 135}

casas_base_config = {'dataset': 'casas', 'base_config': 'context_configs/casas.json',
                     'lag_parameter': 5, 'merge_mins': 1, 'max_time_interval_in_mins': 30, 'input_size': 105}

data_configs = []

for filepath in glob.glob('datasets/incremental_data/*.csv'):
    if 'casas' in filepath:
        casas_config = deepcopy(casas_base_config)
        casas_config['datafile'] = filepath
        data_configs.append(casas_config)
    elif 'extrasensory' in filepath:
        extra_config = deepcopy(extrasensory_base_config)
        extra_config['datafile'] = filepath
        data_configs.append(extra_config)
model_configs = [
    {'model_re': 'TAE', 'stacked_input': False},
]

global_config = {
    'device': 'cuda:1',
    "parse_style": "incremental",
    "access_cache": False,
    'sliding_parameter': 1,
    "cache_dir": "cache/rq1b_inc",
    "log_dir": "cache/logs/rq1b_inc",
    'model_cluster': 'DCN',
    'num_epochs_ae': 300,
    'num_epochs_cluster': 100,
    'reconstruction_conf_val': 0.5,
    'lr_ae': 1e-2,
    'lr_cluster': 1e-3,
    'cnet_type': 'kmeans',
    "fcn_hidden_size": 256,
    'cnet_n_clusters': 30,
    'cnet_cluster_loss_ratio': 0.8,
    'kmeans_min_clusters': 10,
    'kmeans_max_clusters': 30,
    'model_labeler': 'onto_conv',
}


def get_configs():
    final_configs = []
    for dconfig in data_configs:
        for mconfig in model_configs:
            new_config = deepcopy(global_config)
            new_config.update(dconfig)
            new_config.update(mconfig)
            datainfo = dconfig['datafile'].split("/")[-1].split(".")[0]
            experiment_name = f"{datainfo}_{dconfig['lag_parameter']}_"
            experiment_name += f"{dconfig['merge_mins']}_{mconfig['model_re']}"
            new_config['experiment'] = experiment_name
            final_configs.append(new_config)
    return final_configs
