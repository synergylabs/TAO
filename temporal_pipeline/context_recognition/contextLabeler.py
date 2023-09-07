'''
This is wrapper class for all kind of context labeling across datasets
'''

class contextLabeler:

    def __init__(self, labeler_model, run_config, logger):
        self.config = run_config
        self.logger = logger
        self.labeler = labeler_model(run_config, logger)

    def label_clusters(self, context_representations):
        self.labeler.label_clusters(context_representations)
        return None

    def get_cluster_label(self,cluster_id:int):
        '''
        Get cluster label based on id
        :param cluster_id: id of cluster
        :return:
        '''
        return self.labeler.get_cluster_label(cluster_id)

    def get_all_labels(self):
        return self.labeler.cluster_labels


    def get_ontolist(self):
        return self.labeler.ontolist

    def load(self):
        return self.labeler.load()

    def save(self):
        self.labeler.save()
        return None




