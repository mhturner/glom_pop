import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data
from scipy.signal import resample
import pandas as pd
from scipy.stats import ttest_1samp

from glom_pop import dataio, util

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
data_directory = os.path.join(sync_dir, 'datafiles')
video_dir = os.path.join(sync_dir, 'behavior_videos')
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# 2022-03-24.hdf5 : 1, 7, 12, 16
# 2022-03-18.hdf5: 8
datasets = [('20220318', 8),
            ('20220324', 1),
            ('20220324', 7),
            ('20220324', 12),
            ('20220324', 16),
            ]

corr_with_running = []
for ds in datasets:
    series_number = ds[1]
    file_name = '{}-{}-{}.hdf5'.format(ds[0][:4], ds[0][4:6], ds[0][6:])

    # For video:
    series_dir = 'series' + str(series_number).zfill(3)
    date_dir = ds[0]
    file_path = os.path.join(data_directory, file_name)
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get video data:
    # Timing command from voltage trace
    voltage_trace, _, voltage_sample_rate = ID.getVoltageData()
    frame_triggers = voltage_trace[0, :]  # First voltage trace is trigger out

    video_filepath = glob.glob(os.path.join(video_dir, date_dir, series_dir) + "/*.avi")[0]  # should be just one .avi in there
    video_results = dataio.get_ball_movement(video_filepath,
                                             frame_triggers,
                                             sample_rate=voltage_sample_rate)
    fh, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(video_results['frame'], cmap='Greys_r')
    ax[1].imshow(video_results['cropped_frame'], cmap='Greys_r')

    fh, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(video_results['frame_times'], video_results['rmse'], 'k')

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    # Resample to imaging rate and plot

    err_rmse_ds = resample(video_results['rmse'], response_data.get('response').shape[1])  # DO this properly based on response

    # Align running responses
    _, running_response_matrix = ID.getEpochResponseMatrix(err_rmse_ds[np.newaxis, :], dff=False)
    eg_trials = np.arange(0, 50)

    concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in range(epoch_response_matrix.shape[1])], axis=1)
    concat_running = np.concatenate([running_response_matrix[:, x, :] for x in range(running_response_matrix.shape[1])], axis=1)
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')

    fh, ax = plt.subplots(1+len(included_gloms), 1, figsize=(12, 8))
    [x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
    [util.clean_axes(x) for x in ax.ravel()]
    [x.set_ylim() for x in ax.ravel()]

    ax[0].plot(concat_running[0, :], color='k')
    ax[0].set_ylim([err_rmse_ds.min(), 40])
    ax[0].set_ylabel('Movement', rotation=0)
    for g_ind, glom in enumerate(included_gloms):
        ax[1+g_ind].set_ylabel(glom)
        ax[1+g_ind].plot(concat_response[g_ind, :], color=util.get_color_dict()[glom])
        ax[1+g_ind].set_ylim([-0.1, 0.8])

    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    running_amp = ID.getResponseAmplitude(running_response_matrix, metric='mean')

    new_beh_corr = np.array([np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)


corr_with_running = np.vstack(corr_with_running)  # flies x gloms
# %%
corr_with_running.shape
fh, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.axhline(y=0, color='k', alpha=0.5)
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], 0)

    if t_result.pvalue < 0.05:
        ax.annotate('*', (g_ind, 0.45), fontsize=12)

    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax.plot(g_ind * np.ones(corr_with_running.shape[0]), corr_with_running[:, g_ind],
            marker='.', color='k', linestyle='none', alpha=0.5)

    ax.plot(g_ind, y_mean,
            marker='o', color=util.get_color_dict()[glom])

    ax.plot([g_ind, g_ind], [y_mean-y_err, y_mean+y_err],
            color=util.get_color_dict()[glom])

    ax.set_ylim([-0.6, 0.6])
ax.set_xticks(np.arange(0, len(included_gloms)))
ax.set_xticklabels(included_gloms, rotation=90)
ax.set_ylabel('Corr. with behavior (r)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# %%

# Gain of each trial as a function of window average past movement
behavior_amp = []
past_behaviors = []
window_width = 1.0  # sec
stim_timing = ID.getStimulusTiming()
stim_starts = stim_timing['stimulus_start_times']
stim_ends = stim_timing['stimulus_end_times']
stim_middle_times = stim_starts + (stim_ends-stim_starts)/2
stim_middle_times
fh, ax = plt.subplots(10, 10, figsize=(12, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
[x.set_ylim([-0.1, 0.75]) for x in ax]
for t, stim_start in enumerate(stim_middle_times):
    st = np.where(video_results['frame_times'] > (stim_start-window_width))[0][0]
    ed = np.where(video_results['frame_times'] < (stim_start))[0][-1]
    past_behavior = video_results['rmse'][st:ed]
    past_behaviors.append(past_behavior)
    # weighed_average_behavior = np.average(past_behavior,
    #                                       weights=np.linspace(0, 1, len(past_behavior)))
    weighed_average_behavior = np.average(past_behavior)
    behavior_amp.append(weighed_average_behavior)

    for g_ind, glom in enumerate(included_gloms):
        ax[t].plot(epoch_response_matrix[g_ind, t, :], color=util.get_color_dict()[glom], alpha=0.5)
    ax_inset = ax[t].inset_axes([0.1, 0.4, 0.25, 0.50])
    ax_inset.set_axis_off()
    ax_inset.set_ylim([0, 60])
    ax_inset.plot(past_behavior, 'k')

behavior_amp = np.array(behavior_amp)
# past_behaviors = np.vstack(past_behaviors)

# %%






# %% response - triggered walking?

# %% summary








# %%
