'''
This is main script to train context sensing based on config
'''

# core libraries
import faulthandler

faulthandler.enable()

# custom libraries
from context_recognition.dataparsers import *
from context_v0.dataconfigs import *
from context_v0.temporal_clustering.deep_temporal_clustering.DTC import DeepTemporalClustering
from context_recognition.temporal_clustering.deep_temporal_clustering.utils import get_arguments

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
# get parameters for DTC Model
out_dir = f'cache/dtc/extrasensory_' \
          f'{extraSensoryConfig.lag_parameter}_{extraSensoryConfig.sliding_parameter}_' \
          f'{extraSensoryConfig.max_time_interval_in_mins}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

dtc_parser = get_arguments(cnn_kernel_size=len(extraSensoryConfig.activity_labels),
                           pool=5, lr_ae=1e-3, epochs_ae=50, epochs_cluster=50,similarity='EUC',
                           weights_path=f'{out_dir}/weights', n_clusters=15)
dtc_args = dtc_parser.parse_args()
if not os.path.exists(dtc_args.path_weights):
    os.makedirs(dtc_args.path_weights)

# Add additional argument for dtc model
dtc_args.path_weights_ae = os.path.join(dtc_args.path_weights, "autoencoder_weight.pth")
dtc_args.path_weights_main = os.path.join(
    dtc_args.path_weights, "full_model_weigths.pth"
)
dtc_args.device = "cpu"
dtc_args.input_size = X.shape[1]

# Spawn & Train DTC Model
dtc_ae = DeepTemporalClustering(dtc_args)
dtc_ae.load_dataset(X)
# dtc_ae.pretrain_autoencoder()
dtc_ae.init_ClusterNet()
max_roc_score = dtc_ae.training_function()

print("Finished")
X_t = torch.from_numpy(X.astype(np.float32))
_, _, cluster_probabilities, _ = dtc_ae.model(X_t)
Z_contexts = dtc_ae.get_centroids()
X_contexts = dtc_ae.get_reconstructed_input(Z_contexts)

num_activities = len(extraSensoryConfig.activity_labels)
context_sets = []
for cluster_idx in range(X_contexts.shape[0]):
    activity_set_seq = np.array([
        X_contexts[cluster_idx, :][start_idx:start_idx + num_activities] for start_idx in
        range(0, X_contexts.shape[1], num_activities)])
    cluster_rep = {}
    for activity_set_id in range(activity_set_seq.shape[0]):
        activity_set = np.array(extraSensoryConfig.activity_labels)[
            np.where(np.array(activity_set_seq[activity_set_id], dtype=int))[0]]
        for activity in activity_set:
            if activity not in cluster_rep.keys():
                cluster_rep[activity] = 0
            cluster_rep[activity]+=1
    context_sets.append(cluster_rep)

# write trained model_set to cache
cache_dir = 'cache/trained_models/dtc'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_cache_file = f'{cache_dir}/extrasensory_{datetime.now().strftime("%m-%d_%H%M%S")}'
model_cache = {
    'dtc_model': dtc_ae,
    'Z_contexts': Z_contexts,
    'context_sets': context_sets,
}
pickle.dump(model_cache, open(model_cache_file, 'wb'))
# print(np.unique(Z_labels).shape, cnn_ae.model.components_)
