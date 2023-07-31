from copy import deepcopy
import glob


realworld_base_config = {'dataset': 'realworld', 'base_config': 'context_configs/realworld.json',
 'lag_parameter': 0.05, 'merge_mins': 0.01, 'max_time_interval_in_mins': 1}


data_configs = []

for filepath in glob.glob('datasets/realworld_lopo_mites/p7*'):
    realworld_config = deepcopy(realworld_base_config)
    realworld_config['datafile'] = filepath
    data_configs.append(realworld_config)

model_configs = [
    {'model_re': 'TAE', 'stacked_input': False},
]

global_config = {
    'device': 'cuda:1',
    "parse_style": "combined",
    # "access_cache": False,
    'sliding_parameter': 1,
    "cache_dir": "cache/rq3_rw_mites",
    "log_dir": "cache/logs/rq3_rw_mites",
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
    'kmeans_max_clusters': 20,
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
