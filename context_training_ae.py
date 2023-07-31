'''
This is main script to train context sensing based on config
'''

# core libraries
from sklearn.manifold import TSNE
import faulthandler

faulthandler.enable()

# custom libraries
from context_recognition.dataparsers import *
from context_v0.dataconfigs import *
from context_recognition.temporal_clustering.autoenc_cluster.AE import AE
from context_recognition.temporal_clustering.autoenc_cluster.cluster import KmeansCluster, DBSCANCluster, OPTICSCluster
from context_recognition.temporal_clustering.autoenc_cluster.cluster import plot_clusters

# rawdatafile = 'datasets/aruba_dataset.csv'
# num_ae_epochs = 2
# # Get data into required format based on config values
# X = parse_aruba_dataset(rawdatafile, arubaConfig)

# extrasensory
if True:
    rawdatafile = 'datasets/extrasensory_dataset.csv'
    num_ae_epochs = 80
    # Get data into required format based on config values
    X = parse_extrasensory_dataset(rawdatafile, extraSensoryConfig)
    # X = np.unique(X, axis=0)

# create AE model

cnn_ae = AE(X.shape[1], embedding_size=32, l1_kernel_size=len(extraSensoryConfig.activity_labels), lr_ae=1e-3)
# cnn_ae = AE(X.shape[1], embedding_size=32, model_arch='FCN',l1_kernel_size=len(extraSensoryConfig.activity_labels))
# cnn_ae.load_dataset(X)

for epoch in range(1, num_ae_epochs + 1):
    cnn_ae.load_dataset(X)
    cnn_ae.train(epoch)
    cnn_ae.test(epoch)

# get unique embeddings for creating clusters
Z = cnn_ae.get_embedding(X)
Z_unique = np.unique(Z, axis=0)
# Z_unique = pickle.load(open('cache/z_unique.pkl','rb'))

#----------------- KMEANS -----------------
# create clusters based on Z values
cnn_ae_kmeans = KmeansCluster(min_clusters=10, max_clusters=12)
Z_labels_kmeans = cnn_ae_kmeans.fit_predict(Z_unique)
print(np.unique(Z_labels_kmeans).shape, cnn_ae_kmeans.evaluate(Z_unique))
# get context information from clustering
Z_contexts_kmeans = cnn_ae_kmeans.get_context_representations()
X_contexts_kmeans = cnn_ae.get_reconstructed_input(Z_contexts_kmeans)

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

# plot clusters for visualization
data_embedded = TSNE(n_components=2).fit_transform(Z_unique)
plot_clusters(data_embedded, Z_labels_kmeans, name='KMeans')

#----------------- DBSCAN -----------------
cnn_ae_dbscan = DBSCANCluster()
Z_labels_dbscan = cnn_ae_dbscan.fit_predict(Z_unique)

# get context information from clustering
Z_contexts_dbscan = cnn_ae_dbscan.get_context_representations()
X_contexts_dbscan = cnn_ae.get_reconstructed_input(Z_contexts_dbscan)

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

#----------------- OPTICS -----------------
cnn_ae_optics = OPTICSCluster(min_pts_lb=num_activities//2,
                              min_pts_ub=num_activities,
                              max_eps=np.sqrt(X.shape[1])/4,
                              validation_metric=None)
Z_labels_optics = cnn_ae_optics.fit_predict(Z_unique)

# get context information from clustering
Z_contexts_optics = cnn_ae_optics.get_context_representations()
X_contexts_optics = cnn_ae.get_reconstructed_input(Z_contexts_optics)

num_activities = len(extraSensoryConfig.activity_labels)


context_sets_optics = []
for cluster_idx in range(X_contexts_optics.shape[0]):
    activity_set_seq = np.array([
        X_contexts_optics[cluster_idx, :][start_idx:start_idx + num_activities] for start_idx in
        range(0, X_contexts_optics.shape[1], num_activities)])
    cluster_rep = {}
    for activity_set_id in range(activity_set_seq.shape[0]):
        activity_set = np.array(extraSensoryConfig.activity_labels)[
            np.where(np.array(activity_set_seq[activity_set_id], dtype=int))[0]]
        for activity in activity_set:
            if activity not in cluster_rep.keys():
                cluster_rep[activity] = 0
            cluster_rep[activity]+=1
    context_sets_optics.append(cluster_rep)



# write trained model_set to cache
cache_dir = 'cache/trained_models'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_cache_file = f'{cache_dir}/extrasensory_{datetime.now().strftime("%m-%d_%H%M%S")}'
model_cache = {
    'ae_model': cnn_ae_dbscan,
    'Z_contexts_dbscan': Z_contexts_dbscan,
    'context_sets_dbscan': context_sets_dbscan,
    'Z_contexts_kmeans': Z_contexts_kmeans,
    'context_sets_kmeans': context_sets_kmeans,

}
pickle.dump(model_cache, open(model_cache_file, 'wb'))
print(np.unique(Z_labels_dbscan).shape, cnn_ae_dbscan.model.components_)
