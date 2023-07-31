from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import numpy as np
import os
import joblib
# custom libraries
from utils import CustomClusteringError


class OPTICSCluster:
    def __init__(self, input_size, embedding_size, run_config, logger):
        """

        :param run_config:
        :param logger:
        """
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.config = run_config
        self.logger = logger
        self.min_pts_lb = run_config.optics_min_pts_lb
        self.min_pts_ub = run_config.optics_min_pts_ub
        self.max_eps=run_config.optics_max_eps
        # takes embedding as input
        self.is_input_raw = False
        #todo: Include validation metric for optics clustering
        # if run_config.optics_validation_metric:
        #     self.selection_metric = run_config.optics_validation_metric
        # else:
        self.selection_metric = silhouette_score

    def fit_predict(self,Z):
        """
        Cluster given embeddings using optics cluster
        :param Z:
        :param min_pts:
        :return:
        """
        self.training_samples = Z
        if self.config.optics_min_pts:
            self.min_pts = self.config.optics_min_pts
        else:
            self.min_pts = self.get_optimal_min_pts(Z)

        self.model = OPTICS(min_samples=self.min_pts, max_eps=self.max_eps)
        return self.model.fit_predict(Z)


    def predict(self, Z):
        if self.model:
            return self.model.predict(Z)
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def get_optimal_min_pts(self,Z):
        # iterate linearly and find best points.
        scores = []
        for i in range(self.min_pts_lb, self.min_pts_ub + 1):
            optics_m = OPTICS(min_samples=i,max_eps=self.max_eps)
            predictions = optics_m.fit_predict(Z)
            n_cluster_score = self.selection_metric(Z, predictions)
            print(i, n_cluster_score)
            scores.append(n_cluster_score)
            if self.config.verbose:
                self.logger.info(f"OPTICS clustering score with min pts {i}: {n_cluster_score}")
        optimal_min_pts = np.argmax(scores)
        return optimal_min_pts

    def get_context_representations(self, transform_func=None):
        '''
        Get centroids for learned OPTICS clusters based on transformation function
        :return:
        '''
        # core_points =
        # if transform_func is not None:
        #     transformed_core_points
        labels_ = self.model.labels_
        self.cluster_centers_ = []
        for i in range(np.max(labels_)):
            cluster_core_points = self.training_samples[np.where(labels_ == i)[0]]
            cluster_center = np.mean(cluster_core_points, axis=0)
            self.cluster_centers_.append(cluster_center)
        self.cluster_centers_ = np.array(self.cluster_centers_)
        return self.cluster_centers_

    def save(self):
        model_dir = f'{self.config.cache_dir}/{self.config.experiment}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.model:
            return joblib.dump(self.model,f'{model_dir}/cluster_optics.sav')
        else:
            raise CustomClusteringError("Clustering model not trained...")

    def load(self):
        model_filename = f'{self.config.cache_dir}/{self.config.experiment}/cluster_optics.sav'
        if not os.path.exists(model_filename):
            self.logger.error("OPTICS clustering model does not exist..")
            return False
        else:
            self.model = joblib.load(model_filename)
            return True
