from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
from ast import literal_eval as make_tuple
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LinearRegression

from glom_pop import dataio, util

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'ForestRandomWalk',
                                                                          'include_in_analysis': True})

# %%

all_resp = []
all_concat = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    # ImagingDataObject wants a path to an hdf5 file and a series number from that file
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    epoch_parameters = ID.getEpochParameters()

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom')
    vals, names = dataio.get_glom_mask_decoder(response_data.get('mask'))
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    meanbrain_red = response_data.get('meanbrain')[..., 0]
    meanbrain_green = response_data.get('meanbrain')[..., 1]

    # Align responses
    unique_parameter_values, mean_response, _, _ = ID.getTrialAverages(epoch_response_matrix, parameter_key='current_trajectory_index')
    concatenated_tuning = np.concatenate([mean_response[:, x, :] for x in range(len(unique_parameter_values))], axis=1)  # responses, time (concat stims)

    all_concat.append(concatenated_tuning)
    all_resp.append(mean_response)

all_concat = np.dstack(all_concat)
all_resp = np.stack(all_resp, axis=-1)

# %%

all_concat.shape
all_resp.shape

# %%
stim = 4
fh, ax = plt.subplots(14, 10, figsize=(12, 8))
[x.set_axis_off() for x in ax.ravel()]
[x.set_ylim([-0.12, 0.75]) for x in ax.ravel()]
for i in range(10):
    for g in range(14):
        ax[g, i].plot(all_resp[g, stim, :, i], color=colors[g, :])

# %%

fly_ind = 9

cmap = cc.cm.glasbey
colors = cmap(included_vals/included_vals.max())
fh, ax = plt.subplots(len(included_gloms)+1, len(unique_parameter_values), figsize=(8, 10))
[util.clean_axes(x) for x in ax.ravel()]
fh.subplots_adjust(wspace=0.00, hspace=0.00)
for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, glom in enumerate(included_gloms):
        ax[g_ind+1, u_ind].set_ylim([-0.15, 0.65])
        ax[g_ind+1, u_ind].axhline(0, alpha=0.25, color='k', linewidth=0.5)
        ax[g_ind+1, u_ind].plot(all_resp[g_ind, u_ind, :, fly_ind], color=util.get_color_dict().get(glom), linewidth=2)

        # ax[g_ind, u_ind].plot(all_resp[g_ind, :, u_ind, :], color=colors[g_ind], alpha=0.5)
        if (u_ind == 0):
            ax[g_ind+1, u_ind].set_ylabel(glom, fontsize=12, fontweight='bold')
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
    util.clean_axes(th_ax)
    th_ax.plot(trajectory_time, theta, 'b', label='theta')

# %%

def getBinnedTrajectory(trajectory_time, trajectory_values, sample_time, window_size=0.5):
    mean_vals = np.zeros_like(sample_time)
    mean_vals[:] = np.nan
    for ind_t, t in enumerate(sample_time):
        pull_inds = np.where((trajectory_time >= (t-window_size)) & (trajectory_time < t))[0]
        if len(pull_inds) > 0:
            mean_vals[ind_t] = np.nanmean(trajectory_values[pull_inds])

    return mean_vals


all_resp.shape
all_concat.shape

plt.plot(all_concat[:, :, fly_ind].T)

[all_resp[x, :, :, fly_ind] for x in range(12)][0].shape

tt = np.concatenate([all_resp[x, :, :, fly_ind] for x in range(12)], axis=0).T

plt.plot(tt.T)

X = all_resp[:, :, fly_ind]
X.shape
reg = LinearRegression().fit(X, y)

# %%
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))

included_flies = np.array([2, 3, 5, 6])


cmap = cc.cm.glasbey
colors = cmap(included_vals/included_vals.max())
fh, ax = plt.subplots(len(included_gloms)+1, len(unique_parameter_values), figsize=(10, 8))
[util.clean_axes(x) for x in ax.ravel()]
fh.subplots_adjust(wspace=0.00, hspace=0.00)
for u_ind, un in enumerate(unique_parameter_values):
    for leaf_ind, g_ind in enumerate(leaves):
        name = included_gloms[g_ind]

        ax[leaf_ind+1, u_ind].set_ylim([-0.15, 0.50])
        ax[leaf_ind+1, u_ind].axhline(0, alpha=0.25, color='k', linewidth=0.5)
        ax[leaf_ind+1, u_ind].plot(all_resp[g_ind, u_ind, :, included_flies].mean(axis=0), color=colors[g_ind], linewidth=2)

        # ax[g_ind, u_ind].plot(all_resp[g_ind, :, u_ind, :], color=colors[g_ind], alpha=0.5)
        if (u_ind == 0):
            ax[leaf_ind+1, u_ind].set_ylabel(name, fontsize=11, fontweight='bold', rotation=0)
        if (u_ind == 0) & (g_ind == 0):
            plot_tools.addScaleBars(ax[leaf_ind+1, u_ind], dT=5, dF=0.25, T_value=-0.25, F_value=-0.04)

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
    util.clean_axes(th_ax)
    th_ax.plot(trajectory_time, theta, 'b', label='theta')


fh.savefig(os.path.join(save_directory, 'vr_mean_resp.svg'))

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
[util.clean_axes(x) for x in ax.ravel()]
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
[util.clean_axes(x) for x in ax.ravel()]
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
