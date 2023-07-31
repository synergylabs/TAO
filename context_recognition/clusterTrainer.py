'''
This is wrapper class to train clustering models
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_context('poster')
sns.set_color_codes()


class clusterTrainer:

    def __init__(self, model_cluster, input_size, embedding_size, run_config, logger):

        self.device = run_config.device
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.model = model_cluster(input_size, embedding_size, run_config, logger)
        self.config = run_config
        # takes embedding as input
        self.is_input_raw = self.model.is_input_raw

    def fit_predict(self, Z):
        return self.model.fit_predict(Z)

    def predict(self, Z):
        return self.model.predict(Z)

    def get_context_representations(self):
        return self.model.get_context_representations()

    def get_2D_representation(self, Z):
        data_embedded = TSNE(n_components=2).fit_transform(Z)
        return data_embedded

    def plot_clusters(self, data, labels):
        if not self.data_embedded:
            self.data_embedded = self.get_2D_representation(data)
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        if not self.config.plot_kwds:
            self.config.plot_kwds = {'alpha': 0.1, 's': 80, 'linewidths': 0}

        plt.scatter(data[:, 0], data[:, 1], c=colors, **self.config.plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found.'.format(self.config.experiment), fontsize=24)
        plt.show()
        return None

    def save(self):
        self.model.save()

    def load(self):
        return self.model.load()
