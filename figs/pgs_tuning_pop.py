from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, leaves_list

from glom_pop import dataio, util

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'


# %% PLOT MEAN RESPONSES TO TUNING SUITE

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

all_responses = []
vox_per_glom = []
response_amplitudes = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)
    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    # voxels per glom
    vox_per_glom.append([np.sum(response_data.get('mask') == mv) for mv in vals])

    # Only select gloms in included_gloms
    erm = []
    included_vals = []
    for g_ind, name in enumerate(included_gloms):
        pull_ind = np.where(name == names)[0][0]
        erm.append(response_data.get('epoch_response')[pull_ind, :, :])
        included_vals.append(vals[pull_ind])
    epoch_response_matrix = np.stack(erm, axis=0)
    included_vals = np.array(included_vals)

    cmap = cc.cm.glasbey
    colors = cmap(included_vals/included_vals.max())

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, trial_response_by_stimulus = ID.getMeanBrainByStimulus(epoch_response_matrix)
    n_stimuli = mean_voxel_response.shape[2]

    all_responses.append(mean_voxel_response)
    response_amplitudes.append(response_amp)

# Stack accumulated responses
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)

# Exclude last two stims (full field flashes)
unique_parameter_values = unique_parameter_values[:-2]
all_responses = all_responses[:, :, :-2, :]
response_amplitudes = response_amplitudes[:, :-2, :]

mean_responses = np.mean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.std(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)
std_responses = np.std(all_responses, axis=-1) # (glom, time, param)
vox_per_glom = np.stack(vox_per_glom, axis=-1)

mean_responses.shape
np.save(os.path.join(save_directory, 'mean_chat_responses.npy'), mean_responses)
np.save(os.path.join(save_directory, 'included_gloms.npy'), included_gloms)
np.save(os.path.join(save_directory, 'colors.npy'), colors)

# %% PLOTTING

# # # Cluster on concat responses # # #
# Concatenated response traces
mean_responses = np.mean(all_responses, axis=-1)  # shape=(glom, time, stim, fly)
mean_cat_responses = np.vstack(np.concatenate([mean_responses[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
mean_cat_responses = zscore(mean_cat_responses, axis=1)
mean_cat_responses = pd.DataFrame(mean_cat_responses, index=included_gloms)

# Don't collapse clusters here. Compute the full linkage matrix to make dendrogram
clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='average', distance_threshold=0)
clustering.fit(mean_cat_responses)

# compute linkage matrix, Z:
# Ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
#   create the counts of samples under each node
counts = np.zeros(clustering.children_.shape[0])
n_samples = len(clustering.labels_)
for i, merge in enumerate(clustering.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

Z = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

# DENDROGRAM
fh0, ax0 = plt.subplots(1, 1, figsize=(1.5, 6))
D = dendrogram(Z, p=12, truncate_mode='lastp', ax=ax0,
               above_threshold_color='black',
               color_threshold=1, orientation='left',
               labels=mean_cat_responses.index)
leaves = leaves_list(Z)  # glom indices

# ax0.set_yticks([])
ax0.set_xticks([])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.invert_yaxis()

# Plot mean responses

fh1, ax1 = plt.subplots(len(included_gloms), len(unique_parameter_values), figsize=(10, 6))
[util.cleanAxes(x) for x in ax1.ravel()]

fh1.subplots_adjust(wspace=0.00, hspace=0.00)

for u_ind, un in enumerate(unique_parameter_values):
    for leaf_ind, g_ind in enumerate(leaves):
        name = included_gloms[g_ind]
        if (leaf_ind == 0) & (u_ind == (len(unique_parameter_values)-1)):
            plot_tools.addScaleBars(ax1[leaf_ind, u_ind], dT=-2, dF=0.25, T_value=response_data.get('time_vector')[-1], F_value=-0.08)
        # if (u_ind == 0):
        #     ax1[leaf_ind, u_ind].set_ylabel(name, fontsize=11, rotation=0)
        ax1[leaf_ind, u_ind].plot(response_data.get('time_vector'), mean_responses[g_ind, :, u_ind],
                                  color=colors[g_ind, :], alpha=1.0, linewidth=1.5)
        ax1[leaf_ind, u_ind].fill_between(response_data.get('time_vector'),
                                          mean_responses[g_ind, :, u_ind] - sem_responses[g_ind, :, u_ind],
                                          mean_responses[g_ind, :, u_ind] + sem_responses[g_ind, :, u_ind],
                                          color=colors[g_ind, :], alpha=0.5, linewidth=0)


[x.set_ylim([mean_responses.min(), 0.8]) for x in ax1.ravel()]

fh0.savefig(os.path.join(save_directory, 'pgs_tuning_dendrogram.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'pgs_mean_tuning.svg'), transparent=True)

np.save(os.path.join(save_directory, 'cluster_leaves_list.npy'), leaves)

# %% For example stims, show individual fly responses
# cv := std across animals normalized by mean across animals and across all stims for that glom
# cv = np.std(response_amplitudes, axis=-1) / np.mean(response_amplitudes, axis=(1, 2))[:, None]

# Normalize sigma by the maximum response of this glom across all stims
scaling = np.max(np.mean(response_amplitudes, axis=-1), axis=-1)
cv = np.std(response_amplitudes, axis=-1) / scaling[:, None]
eg_leaf_inds = [3, 6, 11]
eg_stim_ind = 6
fh2, ax2 = plt.subplots(len(eg_leaf_inds), all_responses.shape[-1], figsize=(4, 2.5))

[x.set_ylim([-0.1, 1.2]) for x in ax2.ravel()]
[x.set_axis_off() for x in ax2.ravel()]
for li, leaf_ind in enumerate(eg_leaf_inds):
    g_ind = leaves[leaf_ind]
    for fly_ind in range(all_responses.shape[-1]):
        ax2[li, fly_ind].plot(all_responses[g_ind, :, eg_stim_ind, fly_ind], color='k', alpha=0.5)
        ax2[li, fly_ind].plot(np.mean(all_responses[g_ind, :, eg_stim_ind, :], axis=-1), color=colors[g_ind, :], linewidth=2)
        if fly_ind == 0 & li == 0:
            plot_tools.addScaleBars(ax2[0, 0], dT=2, dF=0.25, T_value=0, F_value=-0.05)

        if fly_ind == 0:
            ax2[li, fly_ind].annotate('cv = {:.2f}'.format(cv[g_ind, eg_stim_ind]), (0, 1.0))

print(unique_parameter_values[eg_stim_ind])
print(np.array(included_gloms)[np.array([leaves[x] for x in eg_leaf_inds])])
fh2.savefig(os.path.join(save_directory, 'pgs_fly_responses.svg'), transparent=True)

fh3, ax3 = plt.subplots(1, 1, figsize=(3, 2.5))
ax3.hist(cv.ravel(), bins=40)
ax3.set_xlabel('Coefficient of Variation')
ax3.set_ylabel('Count');

fh2.savefig(os.path.join(save_directory, 'pgs_fly_responses.svg'), transparent=True)

fh4, ax4 = plt.subplots(1, 1, figsize=(3, 2))
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    mean_cv = np.mean(cv[g_ind, :])
    std_cv = np.std(cv[g_ind, :])
    sem_cv = np.std(cv[g_ind, :]) / np.sqrt(len(cv[g_ind, :]))

    ax4.plot(leaf_ind * np.ones_like(cv[g_ind, :]), cv[g_ind, :], color='k', marker='.', linestyle='none', alpha=0.25, markersize=4)
    ax4.plot(leaf_ind, mean_cv, marker='o', color=colors[g_ind, :])
    ax4.plot([leaf_ind, leaf_ind], [mean_cv-sem_cv, mean_cv+sem_cv], marker='None', color=colors[g_ind, :])

ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels([included_gloms[x] for x in leaves])
ax4.set_ylabel('Coefficient of variation', fontsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11, rotation=90)
ax4.set_ylim([0, cv.ravel().max()])

fh4.savefig(os.path.join(save_directory, 'pgs_fly_cv.svg'), transparent=True)

# %% Trial-to-trial variability


fh5, ax5 = plt.subplots(len(eg_leaf_inds), 5, figsize=(3.0, 2))

[x.set_ylim([-0.2, 1.2]) for x in ax5.ravel()]
[x.set_axis_off() for x in ax5.ravel()]
for li, leaf_ind in enumerate(eg_leaf_inds):
    g_ind = leaves[leaf_ind]

    for trial in range(5):
        ax5[li, trial].plot(trial_response_by_stimulus[eg_stim_ind][g_ind, :, trial], color='k', alpha=0.5)
        ax5[li, trial].plot(np.mean(trial_response_by_stimulus[eg_stim_ind][g_ind, :, :], axis=-1), color=colors[g_ind, :])

        if trial == 0 & li == 0:
            plot_tools.addScaleBars(ax5[0, 0], dT=2, dF=0.25, T_value=0, F_value=-0.08)

fh5.savefig(os.path.join(save_directory, 'pgs_trial_responses.svg'), transparent=True)



# %% Inter-individual correlation for each glom


fh4, ax4 = plt.subplots(1, 1, figsize=(3.0, 2))
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    inter_corr = pd.DataFrame(response_amplitudes[g_ind, :, :]).corr().to_numpy()[np.triu_indices(response_amplitudes.shape[-1], k=1)]
    mean_inter_corr = np.mean(inter_corr)
    std_inter_corr = np.std(inter_corr)
    sem_inter_corr = np.std(inter_corr) / np.sqrt(len(inter_corr))
    # ax4.plot(leaf_ind * np.ones_like(inter_corr), inter_corr, color=colors[g_ind, :], marker='.', linestyle='none', alpha=0.5, markersize=4)
    ax4.plot(leaf_ind, mean_inter_corr, color=colors[g_ind, :], marker='o', markersize=6)
    ax4.plot([leaf_ind, leaf_ind], [mean_inter_corr-sem_inter_corr, mean_inter_corr+sem_inter_corr], marker='None', color=colors[g_ind, :])

ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels([included_gloms[x] for x in leaves])
ax4.set_ylabel('Inter-individual corr. (r)', fontsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11, rotation=90)
ax4.set_ylim([0, 1])

fh4.savefig(os.path.join(save_directory, 'pgs_Inter_Ind_Corr.svg'), transparent=True)


# %% Inter-glom corr

corr_mat = mean_cat_responses.T.corr()

fh4, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(corr_mat, vmin=0, vmax=1.0, cmap='Greys', ax=ax, rasterized=True,
            xticklabels=True, yticklabels=True, cbar_kws={'label': 'Correlation (r)'})

sns.set(font_scale=1.5)
fh4.savefig(os.path.join(save_directory, 'pgs_corrmat.svg'), transparent=True)

corr_mat.to_pickle(os.path.join(save_directory, 'pgs_corrmat.pkl'))
mean_cat_responses.to_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl'))
np.save(os.path.join(save_directory, 'pgs_responsepeaks'), response_amplitudes)  # glom x stim x fly
# %%


# %% Individual splits datafiles
# series = [
#           ('2021-08-11', 10),  # R65B05
#           ]

series = [
          ('2021-08-20', 10),  # LC11
          ('2021-08-25', 5),  # LC11
          ]


split_responses = []
fh, ax = plt.subplots(len(series), len(unique_parameter_values), figsize=(14, 2))
[x.set_axis_off() for x in ax.ravel()]
[x.set_ylim([-0.2, 1.4]) for x in ax.ravel()]
for s_ind, ser in enumerate(series):
    experiment_file_name = ser[0]
    series_number = ser[1]
    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Align responses
    time_vector, voxel_trial_matrix = ID.getTrialAlignedVoxelResponses(ID.getRoiResponses('LC11').get('roi_response')[0], dff=True)
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(voxel_trial_matrix)
    n_stimuli = mean_voxel_response.shape[2]

    for u_ind, un in enumerate(unique_parameter_values[:-2]):
        ax[s_ind, u_ind].plot(time_vector, mean_voxel_response[0, :, u_ind], color='k', alpha=1.0, linewidth=2)

    # concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)

    split_responses.append(response_amp)

split_responses = np.vstack(split_responses)


unique_parameter_values

# %% OTHER STUFF

# %% PCA on response amplitudes of individual gloms

# X: individual gloms x features
# Row order = LCa_fly1, LCa_fly2, ..., LCb_fly1, LCb_fly2, ... LCn_flyn
scatter_colors = np.repeat(colors, response_amplitudes.shape[-1], axis=0)
glom_ids = np.repeat(np.arange(0, 12), response_amplitudes.shape[-1], axis=0)

# X: response traces
all_concat = np.hstack([all_responses[:, :, x, :] for x in range(30)])
X = np.reshape(np.swapaxes(all_concat, 1, -1), (-1, all_concat.shape[1]))

# X: response amplitudes
# X = np.reshape(np.swapaxes(response_amplitudes, 1, -1), (-1, response_amplitudes.shape[1]))

X = zscore(X, axis=1)
print('X = {} (samples, features)'.format(X.shape))
pca = PCA(n_components=30)
pca.fit(X)
fh4, ax4 = plt.subplots(1, 1, figsize=(3, 3))
ax4.plot(np.arange(1, 31), pca.explained_variance_ratio_, 'ko')
ax4.set_xlabel('Mode')
ax4.set_ylabel('Frac. var. explained')

fh5, ax5 = plt.subplots(3, 1, figsize=(4, 3))
ax5[0].plot(pca.components_[0, :], 'r-')
ax5[1].plot(pca.components_[1, :], 'g-')
ax5[2].plot(pca.components_[2, :], 'b-')

x_r = pca.fit_transform(X)
fh6, ax6 = plt.subplots(1, 1, figsize=(5, 5))
ax6.scatter(x_r[:, 0], x_r[:, 1], c=scatter_colors)

x_r.shape
