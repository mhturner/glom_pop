from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
# import colorcet as cc
import pandas as pd
import seaborn as sns
import ants

from glom_pop import dataio, util, alignment

# TODO: cleanup. Check timing

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PGS_Reduced',
                                                                          'include_in_analysis': True})

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

glom_size_threshold = 10


# %% ALL FLIES
all_responses = []
response_amplitudes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    # Align responses
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

    response_amp = ID.getResponseAmplitude(mean_response, metric='max')

    all_responses.append(mean_response)
    response_amplitudes.append(response_amp)

    # Trial by trial variability
    stim_ind = 2
    trial_response_amp = ID.getResponseAmplitude(trial_response_by_stimulus[stim_ind], metric='max')

    fh, ax = plt.subplots(len(included_gloms), 30, figsize=(20, 6))
    [x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
    [util.clean_axes(x) for x in ax.ravel()]
    [x.set_ylim() for x in ax.ravel()]
    for g_ind, glom in enumerate(included_gloms):
        ax[g_ind, 0].set_ylabel(glom)
        for t in range(30):
            ax[g_ind, t].plot(trial_response_by_stimulus[stim_ind][g_ind, t, :], color=util.get_color_dict()[glom])

    print('------------')

# %%

cmats = []
for series in matching_series:
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)
    print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)
    parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
    pull_inds = [np.where([pv == up for pv in parameter_values])[0] for up in unique_parameter_values]
    response_amplitude = ID.getResponseAmplitude(epoch_response_matrix, metric='max')  # gloms x trials
    gain_by_trial = np.zeros_like(response_amplitude)
    for pind in pull_inds:
        gain_by_trial[:, pind] = response_amplitude[:, pind] / np.nanmedian(response_amplitude[:, pind], axis=1)[:, np.newaxis]
    # fh, ax = plt.subplots(len(included_gloms), 1, figsize=(8, 12))
    # [util.clean_axes(x) for x in ax.ravel()]
    # # [x.set_ylim([0, 2.5]) for x in ax.ravel()]
    # colors = 'rgbcmk'
    # for g_ind, glom in enumerate(included_gloms):
    #     ax[g_ind].set_ylabel(glom)
    #     ax[g_ind].axhline(y=1, color='k')
    #     ax[g_ind].plot(gain_by_trial[g_ind, :], linestyle='-', color='k')
    #     # for i, pind in enumerate(pull_inds):
    #     #     ax[g_ind].plot(pind, gain_by_trial[g_ind, pind], marker='.', linestyle='none', color=colors[i])

    cmat = pd.DataFrame(data=gain_by_trial.T).corr().to_numpy()
    cmats.append(cmat)

mean_cmat = np.mean(np.dstack(cmats), axis=-1)
mean_cmat = pd.DataFrame(mean_cmat, index=included_gloms, columns=included_gloms)
# %%
# load transformed atlas and mask
glom_mask_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')).numpy()
template_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'JRC2018_reg2meanbrain.nii'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))

glom_mask_2_meanbrain = alignment.filter_glom_mask_by_name(mask=glom_mask_2_meanbrain,
                                                           vpn_types=vpn_types,
                                                           included_gloms=included_gloms)


mean_cmat.mean()
fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(mean_cmat.replace(1, np.nan), cmap='RdBu_r', vmin=-1, vmax=+1, ax=ax[0])

util.make_glom_map(ax=ax[1],
                   glom_map=glom_mask_2_meanbrain,
                   z_val=None,
                   highlight_names='all',
                   colors='glasbey')
# %%
mean_by_stim = [np.nanmean(response_amplitude[:, pi], axis=1) for pi in pull_inds]
std_by_stim = [np.nanstd(response_amplitude[:, pi], axis=1) for pi in pull_inds]
median_by_stim = [np.nanmedian(response_amplitude[:, pi], axis=1) for pi in pull_inds]

modulation_index = [response_amplitude[:, pull_inds[pind]] / mean_by_stim[pind][:, np.newaxis] for pind in range(len(pull_inds))]
# modulation_index shape = (gloms, trials, stims)
modulation_index = np.dstack(modulation_index)
fh, ax = plt.subplots(len(included_gloms), 6, figsize=(8, 12))
[util.clean_axes(x) for x in ax.ravel()]
[x.set_ylim([0, 2.5]) for x in ax.ravel()]
colors = 'rgbcmk'
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for s in range(6):
        ax[g_ind, s].axhline(y=1, color='k')
        ax[g_ind, s].plot(modulation_index[g_ind, :, s], linestyle='-', color=colors[s])


# %%
fh, ax = plt.subplots(len(included_gloms), 6, figsize=(8, 8))
[util.clean_axes(x) for x in ax.ravel()]
colors = 'rgbcmk'
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for s in range(6):
        ax[g_ind, s].hist(modulation_index[g_ind, :, s])
        # ax[g_ind, s].set_xlim([0, 3])
        ax[g_ind, s].axvline(x=1, color='k')

# %%

eg_trials = np.arange(0, 80)
fh, ax = plt.subplots(len(included_gloms), len(eg_trials), figsize=(16, 12))

[x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
[x.set_ylim([-0.1, 1.0]) for x in ax.ravel()]
colors = 'rgbcmk'

for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    ct = 0
    for t in range(epoch_response_matrix.shape[1]):
        stim_ind = np.where([up == parameter_values[t] for up in unique_parameter_values])[0][0]
        if t in eg_trials:
            color = colors[stim_ind]
            ax[g_ind, ct].plot(mean_response[g_ind, stim_ind, :], color=[0.5, 0.5, 0.5])
            ax[g_ind, ct].plot(epoch_response_matrix[g_ind, t, :], color=color)
            ct += 1





# %% PGS_reduced
matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PGS_Reduced',
                                                                          'include_in_analysis': True})

series = matching_series[0]
series_number = series['series']
file_path = series['file_name'] + '.hdf5'
print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)


response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

# epoch_response_matrix: shape=(gloms, trials, time)
epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
epoch_response_matrix[:] = np.nan

for val_ind, included_val in enumerate(included_vals):
    new_glom_size = np.sum(response_data.get('mask') == included_val)

    if new_glom_size > glom_size_threshold:
        pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
        epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
    else:  # Exclude because this glom, in this fly, is too tiny
        pass

# Align responses
unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

response_amp = ID.getResponseAmplitude(mean_response, metric='max')


included_gloms
ep_1 = [x.get('current_diameter', 0) for x in ID.getEpochParameterDicts()]
ep_2 = [x.get('current_intensity', 0) for x in ID.getEpochParameterDicts()]

fh, ax = plt.subplots(5, 20, figsize=(20, 5))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.10, 1.25]) for x in ax]
for t in range(100):
    color = 'k'
    if ep_1[t] == 15:
        if ep_2[t] == 0:
            color='r'
    ax[t].plot(epoch_response_matrix[0, t, :], color=color)
    ax[t].set_title('{}:{}'.format(int(ep_1[t]), int(ep_2[t])), color=color)

# %% Test the same w old PGS data...


matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                                          'include_in_analysis': True})
series = matching_series[9]
series_number = series['series']
file_path = series['file_name'] + '.hdf5'
print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)


response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

# epoch_response_matrix: shape=(gloms, trials, time)
epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
epoch_response_matrix[:] = np.nan

for val_ind, included_val in enumerate(included_vals):
    new_glom_size = np.sum(response_data.get('mask') == included_val)

    if new_glom_size > glom_size_threshold:
        pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
        epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
    else:  # Exclude because this glom, in this fly, is too tiny
        pass

# Align responses
unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

ep_1 = [x.get('current_diameter', 0) for x in ID.getEpochParameterDicts()]
ep_2 = [x.get('current_intensity', 0) for x in ID.getEpochParameterDicts()]
fh, ax = plt.subplots(5, 20, figsize=(20, 5))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.10, 1.25]) for x in ax]
for t in range(100):
    color = 'k'
    if ep_1[t] == 15:
        if ep_2[t] == 0:
            color='r'
    ax[t].plot(epoch_response_matrix[0, t, :], color=color)
    ax[t].set_title('{}:{}'.format(int(ep_1[t]), int(ep_2[t])), color=color)


# %%
# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

# %%

# %%
response_amplitudes.shape

fh, ax = plt.subplots(len(included_gloms), len(unique_parameter_values), figsize=(6, 7))
[x.set_ylim([-0.15, 0.6]) for x in ax.ravel()]
# [x.set_axis_off() for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for u_ind, up in enumerate(unique_parameter_values):
        ax[0, u_ind].set_title(up)
        ax[g_ind, u_ind].plot(mean_responses[g_ind, u_ind, :], color=util.get_color_dict()[glom])
        # ax[g_ind, u_ind].plot(all_responses[g_ind, u_ind, :, :], alpha=0.5, color='k')

# %%

trial_response_amp = ID.getResponseAmplitude(trial_response_by_stimulus[2], metric='max')

unique_parameter_values
trial_response_by_stimulus[2].shape
stim_ind = 2
g_ind = 7

fh, ax = plt.subplots(len(included_gloms), 30, figsize=(20, 6))
[x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
[x.set_ylim() for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for t in range(30):
        ax[g_ind, t].plot(trial_response_by_stimulus[stim_ind][g_ind, t, :], color=util.get_color_dict()[glom])



# %%
corr = pd.DataFrame(data=trial_response_amp.T).corr()

sns.heatmap(corr)
# %%
