'''
This is main script to train context sensing based on config
'''

# core libraries
import faulthandler

faulthandler.enable()

# custom libraries
from context_recognition.dataparsers import *
from context_v0.dataconfigs import *
from context_recognition.temporal_clustering.autoenc_cluster.cluster import KmeansCluster, DBSCANCluster

# rawdatafile = 'datasets/aruba_dataset.csv'
# num_ae_epochs = 2
# # Get data into required format based on config values
# X = parse_aruba_dataset(rawdatafile, arubaConfig)

# extrasensory
if True:
    rawdatafile = 'datasets/extrasensory_dataset.csv'
    # Get data into required format based on config values
    X = parse_extrasensory_dataset(rawdatafile, extraSensoryConfig)
    X = np.unique(X, axis=0)



#----------------- KMEANS -----------------
# create clusters based on Z values
simple_kmeans = KmeansCluster(min_clusters=10, max_clusters=12)
X_labels_kmeans = simple_kmeans.fit_predict(X)
print(np.unique(X_labels_kmeans).shape, simple_kmeans.evaluate(X))

# get context information from clustering
X_contexts_kmeans = simple_kmeans.get_context_representations()
X_contexts_kmeans[X_contexts_kmeans>0.5] = 1
X_contexts_kmeans[X_contexts_kmeans<0.5] = 0

num_activities = len(extraSensoryConfig.activity_labels)

context_sets_kmeans = []
for cluster_idx in range(X_contexts_kmeans.shape[0]):
    activity_set_seq = np.array([
        X_contexts_kmeans[cluster_idx, :][start_idx:start_idx + num_activities] for start_idx in
        range(0, X_contexts_kmeans.shape[1], num_activities)])
    cluster_rep = {}
    for activity_set_id in range(activity_set_seq.shape[0]):
        activity_set = np.array(extraSensoryConfig.activity_labels)[
            np.where(np.array(activity_set_seq[activity_set_id], dtype=int))[0]]
        for activity in activity_set:
            if activity not in cluster_rep.keys():
                cluster_rep[activity] = 0
            cluster_rep[activity]+=1
    context_sets_kmeans.append(cluster_rep)


#----------------- DBSCAN -----------------
simple_dbscan = DBSCANCluster()
Z_labels_dbscan = simple_dbscan.fit_predict(X)

# get context information from clustering
X_contexts_dbscan = simple_dbscan.get_context_representations()

num_activities = len(extraSensoryConfig.activity_labels)

context_sets_dbscan = []
for cluster_idx in range(X_contexts_dbscan.shape[0]):
    activity_set_seq = np.array([
        X_contexts_dbscan[cluster_idx, :][start_idx:start_idx + num_activities] for start_idx in
        range(0, X_contexts_dbscan.shape[1], num_activities)])
    cluster_rep = {}
    for activity_set_id in range(activity_set_seq.shape[0]):
        activity_set = np.array(extraSensoryConfig.activity_labels)[
            np.where(np.array(activity_set_seq[activity_set_id], dtype=int))[0]]
        for activity in activity_set:
            if activity not in cluster_rep.keys():
                cluster_rep[activity] = 0
            cluster_rep[activity]+=1
    context_sets_dbscan.append(cluster_rep)

# context_sets_dbscan.append([
#     activity_vec_to_activities(activity_set, extraSensoryConfig.activity_labels)
#     for activity_set in activity_set_seq
# ])
# write trained model_set to cache
cache_dir = 'cache/trained_models'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_cache_file = f'{cache_dir}/extrasensory_{datetime.now().strftime("%m-%d_%H%M%S")}'
model_cache = {
    'ae_model': simple_dbscan,
    'Z_contexts_dbscan': Z_contexts_dbscan,
    'context_sets_dbscan': context_sets_dbscan,
    'Z_contexts_kmeans': Z_contexts_kmeans,
    'context_sets_kmeans': context_sets_kmeans,

}
pickle.dump(model_cache, open(model_cache_file, 'wb'))
print(np.unique(Z_labels_dbscan).shape, simple_dbscan.model.components_)
