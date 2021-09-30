from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
import pandas as pd
import seaborn as sns

from glom_pop import dataio, util

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'


# %% PLOT MEAN RESPONSES TO TUNING SUITE

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

fh, ax = plt.subplots(len(included_gloms), 30, figsize=(18, 12))
[util.cleanAxes(x) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.00, hspace=0.00)

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
        pull_ind = np.where(name==names)[0][0]
        erm.append(response_data.get('epoch_response')[pull_ind, :, :])
        included_vals.append(vals[pull_ind])
    epoch_response_matrix = np.stack(erm, axis=0)
    included_vals = np.array(included_vals)

    cmap = cc.cm.glasbey
    colors = cmap(included_vals/included_vals.max())

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(epoch_response_matrix)
    n_stimuli = mean_voxel_response.shape[2]

    all_responses.append(mean_voxel_response)
    response_amplitudes.append(response_amp)

    # # To plot individual fly response traces:
    # for u_ind, un in enumerate(unique_parameter_values[:-2]):
    #     for g_ind, name in enumerate(included_gloms):
    #         ax[g_ind, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[g_ind, :, u_ind], color=colors[g_ind, :], alpha=0.25)
    #         ax[g_ind, u_ind].axhline(color='k', alpha=0.25)

all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
mean_responses = np.mean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.std(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)
vox_per_glom = np.stack(vox_per_glom, axis=-1)
response_amplitudes = np.stack(response_amplitudes, axis=-1)

# For display, exclude last two stims (full field flashes)
for u_ind, un in enumerate(unique_parameter_values[:-2]):
    for g_ind, name in enumerate(included_gloms):
        if (g_ind == 0) & (u_ind == (len(unique_parameter_values[:-2])-1)):
            plot_tools.addScaleBars(ax[g_ind, u_ind], dT=-2, dF=0.25, T_value=response_data.get('time_vector')[-1], F_value=-0.08)
        if (u_ind == 0):
            ax[g_ind, u_ind].set_ylabel(name, fontsize=15, fontweight='bold')
        ax[g_ind, u_ind].plot(response_data.get('time_vector'), mean_responses[g_ind, :, u_ind], color=colors[g_ind, :], alpha=1.0, linewidth=2)
        ax[g_ind, u_ind].fill_between(response_data.get('time_vector'),
                                      mean_responses[g_ind, :, u_ind] - sem_responses[g_ind, :, u_ind],
                                      mean_responses[g_ind, :, u_ind] + sem_responses[g_ind, :, u_ind],
                                      color=colors[g_ind, :], alpha=0.5, linewidth=0)


[x.set_ylim([mean_responses.min(), 0.8]) for x in ax.ravel()]

fh.savefig(os.path.join(save_directory, 'mean_tuning.pdf'))

# %%

# extract peak responses for each stim, fly
sample_period = ID.getAcquisitionMetadata().get('sample_period')
pre_frames = int(ID.getRunParameters().get('pre_time') / sample_period)
stim_frames = int(ID.getRunParameters().get('stim_time') / sample_period)

response_peaks = np.max(all_responses[:, pre_frames:(pre_frames+stim_frames), :, :], axis=1)  # glom, param, fly

fh1, ax1 = plt.subplots(len(included_gloms), 1, figsize=(6, 12))
[x.set_axis_off() for x in ax1]
[x.set_ylim([-0.2, 0.8]) for x in ax1]

for g_ind, name in enumerate(included_gloms):

    ax1[g_ind].plot(response_peaks[g_ind, :, :], color=colors[g_ind, :], marker='.', linestyle='none')
    ax1[g_ind].axhline(0, color='k', alpha=0.5)


# %%

fh2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
for g_ind, name in enumerate(included_gloms):
    inter_corr = pd.DataFrame(response_peaks[g_ind, :, :]).corr().to_numpy()[np.triu_indices(7, k=1)]
    ax2.plot(g_ind * np.ones_like(inter_corr), inter_corr, color=colors[g_ind, :], marker='.', linestyle='none', alpha=0.5, markersize=8)
    ax2.plot(g_ind, np.mean(inter_corr), color=colors[g_ind, :], marker='o', markersize=12)

ax2.set_xticks(np.arange(len(included_gloms)))
ax2.set_xticklabels(included_gloms)
ax2.set_ylabel('Inter-individual correlation (r)', fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='x', labelsize=14, rotation=90)

fh2.savefig(os.path.join(save_directory, 'Inter_Ind_Corr.pdf'))

# %%
all_responses.shape
# %% Cluster on concat responses
mean_responses = np.mean(all_responses, axis=-1)
mean_cat_responses = np.vstack(np.concatenate([mean_responses[:, :, x] for x in np.arange(len(unique_parameter_values[:-2]))], axis=1))
mean_cat_responses = pd.DataFrame(mean_cat_responses, index=included_gloms)

sns.set(font_scale=1.5)
g = sns.clustermap(data=mean_cat_responses, col_cluster=False,
                   figsize=(10, 4), cmap='Greys', vmax=0.5, linewidths=0.0, rasterized=True,
                   tree_kws=dict(linewidths=3, colors='k'), yticklabels=True, xticklabels=False, cbar_pos=(0.98, 0.08, 0.025, 0.65))

# g.ax_heatmap.set_xticks([])
[s.set_edgecolor('k') for s in g.ax_heatmap.spines.values()]
[s.set_linewidth(2) for s in g.ax_heatmap.spines.values()]

fh3 = plt.gcf()

fh3.savefig(os.path.join(save_directory, 'pgs_cluster.pdf'))


# %% Inter-glom corr

corr_mat = mean_cat_responses.T.corr()

fh4, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(corr_mat, vmin=0, vmax=1.0, cmap='Greys', ax=ax, rasterized=True,
            xticklabels=True, yticklabels=True, cbar_kws={'label': 'Correlation (r)'})

sns.set(font_scale=1.5)
fh4.savefig(os.path.join(save_directory, 'pgs_corrmat.pdf'))


# %%

tmp = np.concatenate([all_responses[:, :, x, :] for x in range(all_responses.shape[2])], axis=1)
concat_responses = np.concatenate([tmp[:, :, x] for x in range(tmp.shape[2])]) # ind glom (n gloms x n flies) x concat time


ids = np.tile(np.arange(len(included_gloms)), tmp.shape[2])
concat_responses = pd.DataFrame(concat_responses, index=ids)

glom_colors = [colors[i] for i in ids]

sns.clustermap(data=concat_responses, col_cluster=False, row_cluster=True,
               figsize=(8, 6), cmap='magma', row_colors=glom_colors, method='weighted')
# %%


# %%

pd.DataFrame(np.vstack([names.values, vox_per_glom.mean(axis=-1)]).T)
# %% Individual splits datafiles
# series = [
#           ('2021-08-11', 10),  # R65B05
#           ]

series = [
          ('2021-08-20', 10),  # LC11
          ('2021-08-25', 5),  # LC11
          ]


split_responses = []
fh, ax = plt.subplots(len(series), len(unique_parameter_values[:-2]), figsize=(14, 2))
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
