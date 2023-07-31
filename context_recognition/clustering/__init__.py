from .OPTICSCluster import OPTICSCluster
from .KmeansCluster import KmeansCluster
from .DBSCANCluster import DBSCANCluster
from .HDBSCANCluster import HDBSCANCluster
from .Temporal_ClusterNet import Temporal_ClusterNet
from .DeepClusteringNetwork import DeepClusteringNetwork

def fetch_cluster_model(model_name, logger):
    """
    This function fetches require clustering model based on model name in config
    :param model_name: name of RE model
    :param logger: logging object
    :return:
    """
    model_clustering = None
    if model_name == 'kmeans':
        model_clustering = KmeansCluster
    elif model_name == 'dbscan':
        model_clustering = DBSCANCluster
    elif model_name == 'optics':
        model_clustering = OPTICSCluster
    elif model_name == 'hdbscan':
        model_clustering = HDBSCANCluster
    elif model_name == 'temporal_clustering':
        model_clustering = Temporal_ClusterNet
    elif model_name =='DCN':
        model_clustering = DeepClusteringNetwork

    if model_clustering is None:
        logger.info(f"Unable to get clustering model {model_name}. Exiting...")
        exit(1)

    return model_clustering



