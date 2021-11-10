from visanalysis.analysis import volumetric_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
from ast import literal_eval as make_tuple
import seaborn as sns
import pandas as pd

from glom_pop import dataio, util


experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'
included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='vr_walk', only_included=True)

all_resp = []
all_concat = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

    # ImagingDataObject wants a path to an hdf5 file and a series number from that file
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    epoch_parameters = ID.getEpochParameters()

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom')

    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    meanbrain_red = response_data.get('meanbrain')[..., 0]
    meanbrain_green = response_data.get('meanbrain')[..., 1]


    # Only select gloms in included_gloms
    erm = []
    included_vals = []
    for g_ind, name in enumerate(included_gloms):
        pull_ind = np.where(name==names)[0][0]
        erm.append(response_data.get('epoch_response')[pull_ind, :, :])
        included_vals.append(vals[pull_ind])
    epoch_response_matrix = np.stack(erm, axis=0)
    included_vals = np.array(included_vals)

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, trial_response_by_stimulus = ID.getMeanBrainByStimulus(epoch_response_matrix, parameter_key='current_trajectory_index')
    n_stimuli = mean_voxel_response.shape[2]
    concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)

    all_concat.append(concatenated_tuning)
    all_resp.append(mean_voxel_response)

all_concat = np.dstack(all_concat)
all_resp = np.stack(all_resp, axis=-1)

all_concat.shape
all_resp.shape

epoch_response_matrix.shape
# %%
cmap = cc.cm.glasbey
colors = cmap(included_vals/included_vals.max())

fh, ax = plt.subplots(len(included_gloms)+1, len(unique_parameter_values), figsize=(8, 10))
[util.cleanAxes(x) for x in ax.ravel()]
fh.subplots_adjust(wspace=0.00, hspace=0.00)
for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(included_gloms):
        ax[g_ind+1, u_ind].set_ylim([-0.15, 0.25])
        ax[g_ind+1, u_ind].axhline(0, alpha=0.25, color='k', linewidth=0.5)
        ax[g_ind+1, u_ind].plot(all_resp[g_ind, :, u_ind, :].mean(axis=-1), color=colors[g_ind], linewidth=2)

        # ax[g_ind, u_ind].plot(all_resp[g_ind, :, u_ind, :], color=colors[g_ind], alpha=0.5)
        if (u_ind == 0):
            ax[g_ind+1, u_ind].set_ylabel(name, fontsize=12, fontweight='bold')
        if (u_ind == 0) & (g_ind == 0):
            plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=5, dF=0.10, T_value=-0.25, F_value=-0.04)

    # plot trajectory for this stim
    query = {'current_trajectory_index': un}
    trials, trial_inds = shared_analysis.filterTrials(response_data.get('epoch_response'), ID, query, return_inds=True)

    x_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_x_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_x_trajectory'))).get('tv_pairs')
    y_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_y_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_y_trajectory'))).get('tv_pairs')
    theta_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_theta_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_theta_trajectory'))).get('tv_pairs')

    trajectory_time = [tv[0] for tv in x_tv] + ID.getRunParameters().get('pre_time')
    x_position = np.array([tv[1] for tv in x_tv])
    y_position = np.array([tv[1] for tv in y_tv])
    theta = np.array([tv[1] for tv in theta_tv])

    ax[0, u_ind].plot(trajectory_time, x_position, 'r', label='X')
    ax[0, u_ind].plot(trajectory_time, y_position, 'g', label='Y')
    th_ax = ax[0, u_ind].twinx()
    util.cleanAxes(th_ax)
    th_ax.plot(trajectory_time, theta, 'b', label='theta')


fh.savefig(os.path.join(save_directory, 'vr_mean_resp.pdf'))

# %% fly-to-fly variability
all_concat.shape

sel_glom = 8
fh, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(all_concat[sel_glom, :, :], alpha=0.5)
ax.plot(all_concat[sel_glom, :, :].mean(axis=-1), 'k-')

# %%

mean_responses = np.mean(all_resp, axis=-1)  # mean across flies
mean_cat_responses = np.vstack(np.concatenate([mean_responses[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
mean_cat_responses = pd.DataFrame(mean_cat_responses, index=included_gloms)

g = sns.clustermap(data=mean_cat_responses, col_cluster=False,
                   figsize=(8, 4), cmap='magma', linewidths=0.0, rasterized=True)
# %%

corr_mat = mean_cat_responses.T.corr()

fh2, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.heatmap(corr_mat, vmin=0, vmax=1.0, cmap='Greys', ax=ax, rasterized=True,
            xticklabels=True, yticklabels=True, cbar_kws={'label': 'Correlation (r)'})
fh2.savefig(os.path.join(save_directory, 'vr_corrmat.pdf'))

# %% Glom responses
fh, ax = plt.subplots(1 + concatenated_tuning.shape[0], len(unique_parameter_values), figsize=(18, 18))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 0.75]) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.05, hspace=0.05)



for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(names):
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[g_ind, :, u_ind], color=colors[g_ind, :])
        ax[g_ind+1, u_ind].axhline(color='k', alpha=0.5)
        if (g_ind == 0) & (u_ind == 0):
            plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=0, F_value=-0.2)

        if u_ind == 0:
            ax[g_ind+1, u_ind].set_ylabel(name)


# %%

# %% STABILITY ACROSS TRIALS

glom_ind = 1
fh, ax = plt.subplots(n_stimuli, 1, figsize=(8, 6))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 0.75]) for x in ax.ravel()]

for s_ind in range(n_stimuli):
    trials = trial_response_by_stimulus[s_ind].shape[2]

    for t in range(trials):
        ax[s_ind].plot(response_data.get('time_vector'), trial_response_by_stimulus[s_ind][glom_ind, :, t], alpha=0.5, color='k')

    ax[s_ind].plot(response_data.get('time_vector'), trial_response_by_stimulus[s_ind][glom_ind, :, :].mean(axis=-1), color='b', alpha=1.0, linewidth=3)
    if (s_ind == 0):
        plot_tools.addScaleBars(ax[s_ind], dT=1, dF=0.50, T_value=-0.1, F_value=-0.2)

# %%

fh, ax = plt.subplots(14, 1, figsize=(16, 8))
for i in range(14):
    ax[i].plot(response_data['response'][i, :])
    ax[i].set_axis_off()
