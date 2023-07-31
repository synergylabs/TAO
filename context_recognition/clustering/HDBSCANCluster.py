from hdbscan import HDBSCAN
from cuml.cluster import HDBSCAN as cuHDBSCAN

from sklearn.metrics import silhouette_score
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as cu_silhouette_score

from sklearn.neighbors import NearestNeighbors
from cuml.neighbors import NearestNeighbors as cuNearestNeighbours

from kneed import KneeLocator
import numpy as np
import os
import joblib

# custom library
from utils import CustomClusteringError


class HDBSCANCluster:
    def __init__(self, input_size, embedding_size, run_config, logger):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.config = run_config
        self.logger = logger
        # takes embedding as input
        self.is_input_raw = False

        if 'cuda' in self.config.device:
            self.selection_metric = cu_silhouette_score
        else:
            self.selection_metric = silhouette_score

    def fit_predict(self, Z):

        min_samples, min_cluster_size, cluster_epsilon = self.select_hyperparameters(Z)
        if 'cuda' in self.config.device:
            self.model = cuHDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                                   cluster_selection_epsilon=cluster_epsilon)
        else:
            self.model = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                                 cluster_selection_epsilon=cluster_epsilon)

        predictions = self.model.fit_predict(Z)
        # self.cluster_centers_ = np.arange(np.max(predictions)) # dummy data
        # labels_ = self.model.labels_
        # probabilities_ = self.model.probabilities_
        # for i in range(max(labels_)+1):
        #     prob_percentile = np.percentile(probabilities_[np.where(labels_ == i)[0]], 50)
        #     cluster_core_points = Z[np.where((labels_ == i) & (probabilities_ >= prob_percentile))[0]]
        #     cluster_center = np.mean(cluster_core_points, axis=0)
        #     self.cluster_centers_.append(cluster_center)
        #     print(f"Cluster Center {i}: {cluster_center}")
        # self.cluster_centers_ = np.array(self.cluster_centers_)

        return predictions

    # def predict(self, Z):
    #     if self.model:
    #         return self.model.predict(Z)
    #     else:
    #         raise CustomClusteringError("Clustering model not trained...")

    def select_hyperparameters(self, Z):
        '''
        Select best hyperparameters based on config
        :return:
        '''
        min_samples = self.config.hdbscan_min_sample_size
        cluster_epsilon = self.get_optimal_eps(Z, min_samples)
        if self.config.verbose:
            self.logger.info(f"HDBSCAN clustering epsilon value: {cluster_epsilon:.4f}")
        scores = []
        for min_cluster_size in self.config.hdbscan_min_cluster_size:
            if 'cuda' in self.config.device:
                hdbscan_m = cuHDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                                      cluster_selection_epsilon=cluster_epsilon)
            else:
                hdbscan_m = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,
                                    cluster_selection_epsilon=cluster_epsilon)
            predictions = hdbscan_m.fit_predict(Z)
            # self.logger.info(hdbscan_m.cluster_persistence_)
            del hdbscan_m
            num_clusters = np.max(predictions)
            sil_score = self.selection_metric(Z, predictions)
            num_cluster_score = np.abs(self.config.hdbscan_ideal_cluster_count - num_clusters)
            total_score = sil_score - (self.config.hdbscan_cluster_count_reg_coeff * num_cluster_score)
            scores.append(total_score)
            if self.config.verbose:
                self.logger.info(
                    f"HDBSCAN clustering with min cluster size {min_cluster_size}-> sil score:{sil_score}, "
                    f"num_clusters:{num_clusters}, total score: {total_score}")
        optimal_min_cluster_size = self.config.hdbscan_min_cluster_size[np.argmax(scores)]
        return min_samples, optimal_min_cluster_size, cluster_epsilon

    def get_optimal_eps(self, Z, min_samples):
        # choose right eps for given embedding structure
        if 'cuda' in self.config.device:
            nbrs = cuNearestNeighbours(n_neighbors=min_samples).fit(Z)
        else:
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(Z)

        distances, _ = nbrs.kneighbors(Z)
        distance_desc = sorted(distances[:, min_samples - 1], reverse=True)[:10000]

        # find knee for distances to find optimal epsilon values
        kneedle = KneeLocator(range(1, len(distance_desc) + 1),
                              distance_desc,
                              S=50,
                              curve="convex",
                              direction="decreasing")

        return kneedle.knee_y / 2

    def get_context_representations(self, transform_func=None):
        '''
        Get centroids for learned DBSCAN clusters based on transformation function
        :return:
        '''

        return self.cluster_centers_

    def save(self):
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.model:
            return joblib.dump(self.model, f'{model_dir}/cluster_hdbscan.sav')
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def load(self):
        model_filename = f'{self.config.cache_dir}/{self.config.experiment}/cluster_hdbscan.sav'
        if not os.path.exists(model_filename):
            self.logger.error("DBSCAN clustering model does not exist..")
            return False
        else:
            self.model = joblib.load(model_filename)
            return True
