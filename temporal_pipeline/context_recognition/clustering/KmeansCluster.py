'''
Main class for clustering using kmeans
'''
import torch.cuda
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score, calinski_harabasz_score
from cuml.cluster import KMeans as cuKmeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_samples as cu_silhouette_samples

import numpy as np
import joblib
import os
import time

# custo libraries
from utils import CustomClusteringError


class KmeansCluster:
    def __init__(self, input_size, embedding_size, run_config, logger):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.config = run_config
        self.logger = logger
        self.min_clusters = run_config.kmeans_min_clusters
        self.max_clusters = run_config.kmeans_max_clusters
        # takes embedding as input
        self.is_input_raw = False
        self.selection_method = run_config.kmeans_selection_metric
        if self.selection_method == 'sil':
            if 'cuda' in self.config.device:
                self.selection_metric = cu_silhouette_samples
            else:
                self.selection_metric = silhouette_score
        elif self.selection_method == 'dbi':
            self.selection_metric = davies_bouldin_score
        elif self.selection_method == 'chs':
            self.selection_metric = calinski_harabasz_score
        else:
            raise CustomClusteringError("Unknown selection method for optimal cluster counting")

    def fit_predict(self,Z):
        self.n_clusters = self.find_optimal_cluster_count(Z)
        if 'cuda' in self.config.device:
            self.model = cuKmeans(n_clusters=self.n_clusters)
        else:
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters)

        return self.model.fit_predict(Z)

    def find_optimal_cluster_count(self,Z):
            scores = []
            self.logger.info("Getting optimum cluster count for Kmeans")
            for i in range(self.min_clusters,self.max_clusters + 1):
                start_time = time.time()
                if 'cuda' in self.config.device:
                    km = cuKmeans(n_clusters=i)
                    predictions = km.fit_predict(Z)
                    n_cluster_score = np.mean(self.selection_metric(Z, predictions, chunksize=10000)).item()
                else:
                    km = MiniBatchKMeans(n_clusters=i)
                    predictions = km.fit_predict(Z)
                    n_cluster_score = self.selection_metric(Z, predictions)
                del km
                self.logger.info(f"{self.selection_method} Score for {i} clusters: {n_cluster_score} in {time.time()-start_time:.3f} secs")
                scores.append(n_cluster_score)
            optimal_cluster_count = np.argmax(scores) + self.min_clusters
            torch.cuda.empty_cache()
            if self.selection_method=='dbi':
                optimal_cluster_count = np.argmin(scores) + self.min_clusters

            return optimal_cluster_count

    def update_selection_method(self,selection_method):
        self.selection_method = selection_method
        if self.selection_method == 'sil':
            self.selection_metric = silhouette_score
        elif self.selection_method == 'dbi':
            self.selection_metric = davies_bouldin_score
        elif self.selection_method == 'chs':
            self.selection_metric = calinski_harabasz_score
        else:
            raise CustomClusteringError("Unknown selection method for optimal cluster counting")

    def predict(self,Z):
        if self.model:
            return self.model.predict(Z)
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def evaluate(self,Z):
        predictions = self.predict(Z)
        return self.selection_metric(Z,predictions)

    def get_context_representations(self):
        return self.model.cluster_centers_

    def save(self):
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.model:
            return joblib.dump(self.model,f'{model_dir}/cluster_kmeans.sav')
        else:
            raise CustomClusteringError("Clustering model not trained...")


    def load(self):
        model_filename = f'{self.config.cache_dir}/{self.config.experiment}/cluster_kmeans.sav'
        if not os.path.exists(model_filename):
            self.logger.error("Kmeans clustering model does not exist..")
            return False
        else:
            self.model = joblib.load(model_filename)
            return True