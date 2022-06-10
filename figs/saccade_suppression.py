from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
from skimage.io import imread
# from flystim import image
import ast
from scipy.interpolate import interp1d


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SaccadeSuppression',
                                                                          'include_in_analysis': True,
                                                                          'saccade_sample_period': 0.25,
                                                                          })

for series in np.array(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('{}: {}'.format(file_name, series_number))

# %%

target_saccade_times = np.arange(0, 3, 0.25)

all_response_gains = []
all_response_amps = []
all_responses = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Load behavior data
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             response_len = response_data.get('response').shape[1],
                                             process_behavior=True, fps=50)
    behaving_trials = np.where(behavior_data.get('is_behaving')[0])[0]
    nonbehaving_trials = np.where(~behavior_data.get('is_behaving')[0])[0]

    trial_averages = np.zeros((len(included_gloms), len(target_saccade_times), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for st_ind, st in enumerate(target_saccade_times):
        _, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                   ID,
                                                                   query={'current_saccade_time': st},
                                                                   return_inds=True)
        behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, st_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, st_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time


    glom_avg_resp = np.mean(trial_averages, axis=(1,2))
    # peak response time, for each glom (across all stim conditions)
    peak_time = response_data['time_vector'][np.argmax(glom_avg_resp, axis=1)] - ID.getRunParameters('pre_time')

    resp_amp = ID.getResponseAmplitude(trial_averages, metric='max')  # glom x saccade time x beh/nonbeh
    # gain := response amp / amp for last saccade time. Norm is done *within each behavioral condition*
    norm_val = resp_amp[:, -1, :]  # last saccade time, basically at the end of the trial, well after the response has ended
    response_gain = resp_amp / norm_val[:, np.newaxis, :]  # gloms x saccade times x beh/nonbeh
    all_response_gains.append(response_gain)
    all_responses.append(trial_averages)
    all_response_amps.append(resp_amp)

    if True: # Plot ind fly responses. QC
        eg_saccade_inds = np.arange(0, len(target_saccade_times), 1)
        fh, ax = plt.subplots(len(included_gloms), len(eg_saccade_inds), figsize=(8, 4))
        fh.suptitle('{}: {}'.format(file_name, series_number))
        [x.set_ylim([-0.1, 0.5]) for x in ax.ravel()]
        [x.set_axis_off() for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            for ind, si in enumerate(eg_saccade_inds):
                ax[g_ind, ind].plot(response_data['time_vector'], trial_averages[g_ind, si, 0, :], color=util.get_color_dict()[glom])
                ax[g_ind, ind].plot(response_data['time_vector'], trial_averages[g_ind, si, 1, :], color=util.get_color_dict()[glom], alpha=0.5)


all_response_gains = np.stack(all_response_gains, axis=-1)
all_response_amps = np.stack(all_response_amps, axis=-1)
all_responses = np.stack(all_responses, axis=-1)
# %% eg glom and fly traces
eg_fly_ind = 4
eg_glom_ind = 0

# mean response trace for timing illustration
eg_saccade_inds = [0, 5, 7, 9, 11]
# eg_saccade_inds = np.arange(12)
tags = 'abcdefghijklmn'
fh0, ax0 = plt.subplots(1, 1, figsize=(2.5, 1.5))
ax0.set_ylim([-0.1, 0.45])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
for ind, si in enumerate(eg_saccade_inds):
    xval = target_saccade_times[si]
    ax0.plot([xval, xval], [0, 0.4], color='k', linestyle='--')
    ax0.annotate(tags[ind], (xval, 0.41), ha='center')
ax0.axhline(y=0, color='k', alpha=0.5)
ax0.set_xlabel('Time (sec)')
ax0.set_ylabel('Response (dF/F)')
ax0.plot(response_data['time_vector']-ID.getRunParameters('pre_time'), np.mean(all_responses[eg_glom_ind, :, 0, :, eg_fly_ind], axis=0),
         color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2, label='behaving')
ax0.plot(response_data['time_vector']-ID.getRunParameters('pre_time'), np.mean(all_responses[eg_glom_ind, :, 1, :, eg_fly_ind], axis=0),
         color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2, label='nonbehaving', alpha=0.5)

fh1, ax1 = plt.subplots(1, len(eg_saccade_inds), figsize=(8, 1.5))
[x.set_ylim([-0.1, 0.5]) for x in ax1.ravel()]
[x.set_axis_off() for x in ax1.ravel()]
for ind, si in enumerate(eg_saccade_inds):
    ax1[ind].axhline(y=0, color='k', alpha=0.5)
    ax1[ind].plot(response_data['time_vector'], all_responses[eg_glom_ind, si, 0, :, eg_fly_ind],
                  color=util.get_color_dict()[included_gloms[eg_glom_ind]])
    ax1[ind].plot(response_data['time_vector'], all_responses[eg_glom_ind, si, 1, :, eg_fly_ind],
                  color=util.get_color_dict()[included_gloms[eg_glom_ind]], alpha=0.5)
    ax1[ind].set_title(tags[ind])
    if ind == 0:
        plot_tools.addScaleBars(ax1[ind], dT=2, dF=0.25, T_value=-0.1, F_value=-0.08)

fh0.savefig(os.path.join(save_directory, 'saccade_timing_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)
fh1.savefig(os.path.join(save_directory, 'saccade_traces_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)

# %% Stim schematic

fh2, ax2 = plt.subplots(1, 1, figsize=(2, 1.5))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tt = np.linspace(0.01, 2.99, 100)
switch_times = [1, 1.5, 2]
for st_ind, st in enumerate(switch_times):
    t = [0, st, st+0.2, 3]
    p = [0, 0, 70, 70]
    interp_fh = interp1d(t, p, kind='linear')
    loc = interp_fh(tt)
    loc -= loc[0]  # start at 0
    ax2.plot(tt, loc, alpha=1, color=[0.5, 0.5, 0.5], linewidth=2)

ax2.quiver(1.6, 35, 1, 0, color='y', scale=3, width=0.02, headwidth=3)
ax2.quiver(1.6, 35, -1, 0, color='y', scale=3, width=0.02, headwidth=3)
ax2.annotate('$\Delta t$', (1.65, 40))
ax2.plot(1.6, 35, 'yo')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Background \nposition ($\degree$)')
fh2.savefig(os.path.join(save_directory, 'saccade_schematic.svg'), transparent=True)


# %% Summary data: probe response gain as a fxn of saccade time

# Response onset := peak slope of mean response across all stims, animals
onset_inds = [np.argmax(np.diff(np.mean(all_responses[x, :, :, :], axis=(0, 2)))) for x in range(len(included_gloms))]

rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

mean_response_gain = np.mean(all_response_gains, axis=-1)
sem_response_gain = np.std(all_response_gains, axis=-1) / np.sqrt(all_response_gains.shape[-1])

fh3, ax3 = plt.subplots(3, 3, figsize=(4.5, 3.5))
[x.set_ylim([0.2, 1.8]) for x in ax3.ravel()]
[x.set_axis_off() for x in ax3.ravel()]
for g_ind, glom in enumerate(included_gloms):
    # indication of response onset time
    tt = response_data['time_vector'] - ID.getRunParameters('pre_time')
    ax3[rows[g_ind], cols[g_ind]].axvline(x=tt[onset_inds[g_ind]],
                                          color='k', alpha=0.5, linestyle=':')

    ax3[rows[g_ind], cols[g_ind]].axhline(y=1, color='k', alpha=0.5)
    ax3[rows[g_ind], cols[g_ind]].errorbar(x=target_saccade_times,
                        y=mean_response_gain[g_ind, :, 0],
                        yerr=sem_response_gain[g_ind, :, 0],
                        color=util.get_color_dict()[glom],
                        linewidth=2, alpha=1)
    ax3[rows[g_ind], cols[g_ind]].errorbar(x=target_saccade_times,
                        y=mean_response_gain[g_ind, :, 1],
                        yerr=sem_response_gain[g_ind, :, 1],
                        color=util.get_color_dict()[glom],
                        linewidth=2, alpha=0.5)
    ax3[rows[g_ind], cols[g_ind]].set_title(glom)



    if g_ind == 5:
        ax3[rows[g_ind], cols[g_ind]].set_axis_on()
        ax3[rows[g_ind], cols[g_ind]].set_ylabel('Visually-driven response gain')
        ax3[rows[g_ind], cols[g_ind]].set_xlabel('Saccade time (s)')
        ax3[rows[g_ind], cols[g_ind]].spines['top'].set_visible(False)
        ax3[rows[g_ind], cols[g_ind]].spines['right'].set_visible(False)
    else:
        ax3[rows[g_ind], cols[g_ind]].set_axis_off()





fh3.savefig(os.path.join(save_directory, 'saccade_gain_summary.svg'), transparent=True)




# %%


# %%
