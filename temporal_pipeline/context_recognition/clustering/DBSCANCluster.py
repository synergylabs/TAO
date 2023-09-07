from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
import numpy as np
import os
import joblib

# custom library
from utils import CustomClusteringError


class DBSCANCluster:
    def __init__(self, input_size, embedding_size, run_config, logger):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.config = run_config
        self.logger = logger

        #takes embedding as input
        self.is_input_raw=False

    def fit_predict(self, Z):
        if (self.config.dbscan_eps) & (self.config.dbscan_min_pts):
            self.eps = self.config.dbscan_eps
            self.min_pts = self.config.dbscan_min_pts
        elif (self.config.dbscan_min_pts):
            self.min_pts = self.config.dbscan_min_pts
            self.eps = self.get_optimal_eps(Z, self.min_pts)
        else:
            self.min_pts = self.get_optimal_min_pts(Z)
            self.eps = self.get_optimal_eps(Z, self.min_pts)

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_pts)
        return self.model.fit_predict(Z)

    def get_optimal_eps(self, Z, min_pts):
        # choose right eps for given embedding structure
        neighbors = min_pts
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(Z)

        distances, indices = nbrs.kneighbors(Z)
        distance_desc = sorted(distances[:, neighbors - 1], reverse=True)[:1000]

        # find knee for distances to find optimal epsilon values
        kneedle = KneeLocator(range(1, len(distance_desc) + 1),
                              distance_desc,
                              S=1.0,
                              curve="convex",
                              direction="decreasing")

        return kneedle.knee_y / 2

    def predict(self, Z):
        if self.model:
            return self.model.predict(Z)
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def get_optimal_min_pts(self, Z):
        # very basic implementation
        return Z.shape[1] // 4

    def get_context_representations(self, transform_func=None):
        '''
        Get centroids for learned DBSCAN clusters based on transformation function
        :return:
        '''
        # core_points =
        # if transform_func is not None:
        #     transformed_core_points
        core_points = self.model.components_
        core_points_labels = self.model.labels_[self.model.core_sample_indices_]
        self.cluster_centers_ = []
        for i in range(np.max(core_points_labels)):
            cluster_core_points = core_points[np.where(core_points_labels == i)[0]]
            cluster_center = np.mean(cluster_core_points, axis=0)
            self.cluster_centers_.append(cluster_center)
        self.cluster_centers_ = np.array(self.cluster_centers_)
        return self.cluster_centers_

    def save(self):
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.model:
            return joblib.dump(self.model,f'{model_dir}/cluster_dbscan.sav')
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def load(self):
        model_filename = f'{self.config.cache_dir}/{self.config.experiment}/cluster_dbscan.sav'
        if not os.path.exists(model_filename):
            self.logger.error("DBSCAN clustering model does not exist..")
            return False
        else:
            self.model = joblib.load(model_filename)
            return True