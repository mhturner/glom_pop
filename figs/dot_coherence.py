from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
import pandas as pd

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

glom_size_threshold = 10

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'CoherentDots',
                                                                          'include_in_analysis': True,
                                                                          # 'speed': [80, -80],
                                                                          # 'speed': 80,
                                                                          # 'coherence': [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0],
                                                                          # 'coherence': [0, 0.25, 0.5, 0.75, 1.0],
                                                                          })


# %%
target_coherence = [0, 0.25, 0.5, 0.75, 1.0]
target_speed = 80
eg_ind = 2

all_responses = []
response_amplitudes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

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

    # Select only trials with target params:
    # Shape = (gloms x param conditions x time)
    trial_averages = np.zeros((len(included_gloms), len(target_coherence), epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    num_matching_trials = []
    for coh_ind, coh in enumerate(target_coherence):
        erm_selected = shared_analysis.filterTrials(epoch_response_matrix,
                                                    ID,
                                                    query={'coherence': coh,
                                                           'speed': target_speed},
                                                    return_inds=False)
        num_matching_trials.append(erm_selected.shape[1])
        trial_averages[:, coh_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x time

    if np.all(np.array(num_matching_trials) > 0):
        print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
        response_amp = ID.getResponseAmplitude(trial_averages, metric='mean')  # shape = gloms x param condition

        all_responses.append(trial_averages)
        response_amplitudes.append(response_amp)

    if s_ind == eg_ind:
        fh0, ax = plt.subplots(len(included_gloms), len(target_coherence), figsize=(3, 6))
        [x.set_ylim([-0.15, 0.5]) for x in ax.ravel()]
        [util.clean_axes(x) for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            ax[g_ind, 0].set_ylabel(glom, fontsize=9)
            for u_ind, up in enumerate(target_coherence):
                if g_ind == 0:
                    ax[0, u_ind].set_title(up)
                    if u_ind == 0:
                        plot_tools.addScaleBars(ax[0, 0], dT=4, dF=0.25, T_value=0, F_value=-0.1)
                ax[g_ind, u_ind].plot(trial_averages[g_ind, u_ind, :], color=util.get_color_dict()[glom])

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

fh0.savefig(os.path.join(save_directory, 'coherence_eg_fly.svg'), transparent=True)

# %% Plot resp. vs. dot coherence
fh1, ax = plt.subplots(len(included_gloms), len(target_coherence), figsize=(3, 6))
[x.set_ylim([-0.15, 0.5]) for x in ax.ravel()]
# [x.set_axis_off() for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for u_ind, up in enumerate(target_coherence):
        if g_ind == 0:
            ax[0, u_ind].set_title(up)
        # ax[g_ind, u_ind].plot(all_responses[g_ind, u_ind, :, :], alpha=0.25, color='k')
        ax[g_ind, u_ind].plot(mean_responses[g_ind, u_ind, :], color=util.get_color_dict()[glom])

fh1.savefig(os.path.join(save_directory, 'coherence_mean_fly.svg'), transparent=True)


# %%
fh2, ax = plt.subplots(1, len(included_gloms), figsize=(6, 2))
[x.set_ylim([-0.1, 0.35]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()[1:]]
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    # Ttest 0 vs. 1 coherence
    h, p = ttest_rel(response_amplitudes[g_ind, 0, :], response_amplitudes[g_ind, 4, :])
    p_vals.append(p)

    if p < 0.05:
        ax[g_ind].annotate('*', (0.5, 0.25), fontsize=18)

    ax[g_ind].axhline(y=0, color='k', alpha=0.50)
    ax[g_ind].set_title(glom, fontsize=9)
    ax[g_ind].plot(target_coherence, response_amplitudes[g_ind, :, :].mean(axis=-1),
                   color=util.get_color_dict()[glom], marker='.', linestyle='-')


    for coh_ind, coh in enumerate(target_coherence):
        mean_val = response_amplitudes[g_ind, coh_ind, :].mean(axis=-1)
        err_val = response_amplitudes[g_ind, coh_ind, :].std(axis=-1) / np.sqrt(response_amplitudes.shape[-1])
        ax[g_ind].plot([coh, coh], [mean_val-err_val, mean_val+err_val],
                       color=util.get_color_dict()[glom], linestyle='-')
    # ax[g_ind].plot(target_coherence, response_amplitudes[g_ind, :, :],
    #                color=util.get_color_dict()[glom], marker='.', linestyle='none', alpha=0.25)

ax[0].set_xlabel('Coherence')
ax[0].set_ylabel('Mean response (dF/F)')
fh2.savefig(os.path.join(save_directory, 'coherence_tuning_curves.svg'), transparent=True)


# %%
response_amplitudes.shape



# %%
