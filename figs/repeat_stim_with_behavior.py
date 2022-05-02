import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp, pearsonr, ttest_rel
from visanalysis.util import plot_tools
from skimage import filters

from glom_pop import dataio, util


# PROTOCOL_ID = 'ExpandingMovingSpot'
PROTOCOL_ID = 'LoomingSpot'

if PROTOCOL_ID == 'ExpandingMovingSpot':
    eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
    target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                              'include_in_analysis': True,
                              'diameter': 15.0,
                              }
    y_min = -0.15
    y_max = 0.80
    eg_trials = np.arange(30, 50)
elif PROTOCOL_ID == 'LoomingSpot':
    # eg_series = ('2022-04-12', 6)  # ('2022-04-12', 2)
    eg_series = ('2022-04-26', 5)  # ('2022-04-12', 2)
    target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                              'include_in_analysis': True,
                              'rv_ratio': 100.0,
                              'center': [0, 0],
                              }
    y_min = -0.05
    y_max = 0.35
    eg_trials = np.arange(25, 45)

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata=target_series_metadata,
                                                  target_groups=['aligned_response', 'behavior'])

# %%
# eg fly figs...
# fh0: snippet of movement and glom response traces
fh0, ax0 = plt.subplots(2+len(included_gloms), 1, figsize=(5.5, 5))
[x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
[util.clean_axes(x) for x in ax0.ravel()]
[x.set_ylim() for x in ax0.ravel()]

# fh1: image of fly on ball
fh1, ax1 = plt.subplots(1, 1, figsize=(1.5, 1.25))
ax1.set_axis_off()
ax1.set_xlim([60, 300])
ax1.set_ylim([200, 0])

# fh2: overall movement trace, with threshold and classification shading
fh2, ax2 = plt.subplots(1, 1, figsize=(3.5, 1.25))
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RMS image \ndifference (a.u.)')

corr_with_running = []

response_amps = []
running_amps = []
all_behaving = []

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response and behavior data
    behavior_data = dataio.load_behavior(ID)
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # downsample behavior from video rate (50 Hz) to imaging rate (~8 Hz)
    behavior_data['rmse'].shape[0] / response_data.get('response').shape[1]
    rmse_ds = resample(behavior_data['rmse'], response_data.get('response').shape[1])
    rmse_ds.shape
    response_data.get('response').shape[1]
    # smooth behavior trace.
    #   Window size about 200 msec, 1st order polynomial
    #   Keeps rapid onsets/offsets pretty well but makes bouts more obvious and continuous
    rmse_ds = savgol_filter(rmse_ds, 9, 1)

    thresh = filters.threshold_li(rmse_ds)
    binary_behavior_ds = (rmse_ds > thresh).astype('int')

    # Align running responses
    _, running_response_matrix = ID.getEpochResponseMatrix(rmse_ds[np.newaxis, :],
                                                           dff=False)
    _, behavior_binary_matrix = ID.getEpochResponseMatrix(binary_behavior_ds[np.newaxis, :],
                                                          dff=False)

    # For running amplitude, take mean rms from start of trial to end of stim time
    pre_frames = int(ID.getRunParameters('pre_time') / ID.getAcquisitionMetadata('sample_period'))
    stim_frames = int(ID.getRunParameters('stim_time') / ID.getAcquisitionMetadata('sample_period'))

    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    running_amp = ID.getResponseAmplitude(running_response_matrix, metric='mean')

    running_amps.append(running_amp)
    response_amps.append(response_amp)

    # Categorize trial as behaving or nonbehaving
    beh_per_trial = np.mean(behavior_binary_matrix[0, :, :], axis=1)
    behaving = beh_per_trial > 0.5  # bool array: n trials
    all_behaving.append(behaving)

    new_beh_corr = np.array([np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    # QC: check thresholding
    # fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].plot(rmse_ds)
    # ax[0].axhline(thresh, color='r')
    # ax[1].hist(rmse_ds, 100)
    # ax[1].axvline(thresh, color='r')
    # ax[0].set_title('{}: {}'.format(file_name, series_number))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([running_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_behaving = np.concatenate([behavior_binary_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_running.shape[1]) * ID.getAcquisitionMetadata('sample_period')

        # Red triangles when stim hits center of screen (middle of trial)
        dt = np.diff(concat_time)[0]  # sec
        trial_len = epoch_response_matrix.shape[2]
        concat_len = len(concat_time)
        y_val = 0.5
        ax0[0].plot(dt * np.linspace(trial_len/2,
                                     concat_len-trial_len/2,
                                     len(eg_trials)),
                    y_val * np.ones(len(eg_trials)),
                    'rv', markersize=4)
        ax0[0].set_ylim([0.25, 0.75])
        ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')
        ax0[0].set_ylabel('Stim', rotation=0)

        ax0[1].plot(concat_time, concat_running[0, :], color='k')
        ax0[1].set_ylim([concat_running.min(), concat_running.max()])
        ax0[1].set_ylabel('Movement', rotation=0)

        for g_ind, glom in enumerate(included_gloms):
            ax0[2+g_ind].set_ylabel(glom, rotation=0)
            ax0[2+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25)
            ax0[2+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[2+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[2+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)

        # Image of fly on ball:
        ax1.imshow(behavior_data['frame'], cmap='Greys_r')

        # Fly movement traj with thresh and binary shading
        tw_ax = ax2.twinx()
        tw_ax.fill_between(behavior_data['frame_times'][:len(behavior_data['binary_behavior'])],
                           behavior_data['binary_behavior'],
                           color=[0.5, 0.5, 0.5], alpha=0.5)
        ax2.axhline(behavior_data['binary_thresh'], color='r')
        ax2.plot(behavior_data['frame_times'][:len(behavior_data['binary_behavior'])],
                 behavior_data['rmse'],
                 'k')
        tw_ax.set_yticks([])

corr_with_running = np.vstack(corr_with_running)  # flies x gloms
running_amps = np.vstack(running_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tw_ax.spines['top'].set_visible(False)
tw_ax.spines['right'].set_visible(False)

fh0.savefig(os.path.join(save_directory, 'repeat_beh_{}_resp.svg'.format(PROTOCOL_ID)), transparent=True)
fh1.savefig(os.path.join(save_directory, 'repeat_beh_{}_flyonball.svg'.format(PROTOCOL_ID)), transparent=True)
fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_running.svg'.format(PROTOCOL_ID)), transparent=True)


# %% Summary plots

# For each fly: corr between trial amplitude and trial behavior amount
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 4.45))
ax2.axvline(0, color='k', alpha=0.50)
ax2.set_xlim([-0.8, 0.8])
ax2.invert_yaxis()

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], 0, nan_policy='omit')
    p_vals.append(t_result.pvalue)

    if t_result.pvalue < (0.05 / len(included_gloms)):
        ax2.annotate('*', (0.80, g_ind), fontsize=12)

    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax2.plot(corr_with_running[:, g_ind], g_ind * np.ones(corr_with_running.shape[0]),
             marker='.', color=util.get_color_dict()[glom], linestyle='none', alpha=0.5)

    ax2.plot(y_mean, g_ind,
             marker='o', color=util.get_color_dict()[glom])

    ax2.plot([y_mean-y_err, y_mean+y_err], [g_ind, g_ind],
             color=util.get_color_dict()[glom])

ax2.set_yticks([])
ax2.set_xlabel('Corr. with behavior (r)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# put diff clusters in rows...
rows = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
cols = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
fh4, ax4 = plt.subplots(4, 4, figsize=(4, 3.5))
[x.set_axis_off() for x in ax4.ravel()]

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    beh = [np.nanmean(response_amps[g_ind, all_behaving[x], x]) for x in range(response_amps.shape[2])]
    nonbeh = [np.nanmean(response_amps[g_ind, np.logical_not(all_behaving[x]), x]) for x in range(response_amps.shape[2])]

    h, p = ttest_rel(beh, nonbeh, nan_policy='omit')
    p_vals.append(p)

    mean_beh = np.nanmean(beh)
    err_beh = np.nanstd(beh) / np.sqrt(len(beh))
    err_beh = np.nanstd(beh)

    mean_nonbeh = np.nanmean(nonbeh)
    err_nonbeh = np.nanstd(nonbeh) / np.sqrt(len(nonbeh))
    err_nonbeh = np.nanstd(nonbeh)

    ax4[rows[g_ind], cols[g_ind]].plot([0, 0.55], [0, 0.55], 'k-', alpha=0.5)
    ax4[rows[g_ind], cols[g_ind]].plot(beh, nonbeh, color='k', alpha=0.5, marker='.', linestyle='none')
    ax4[rows[g_ind], cols[g_ind]].plot(mean_beh, mean_nonbeh,
                                       color=util.get_color_dict()[glom], marker='o', label=glom)

    ax4[rows[g_ind], cols[g_ind]].plot([mean_beh-err_beh, mean_beh+err_beh],
                                       [mean_nonbeh, mean_nonbeh],
                                       color=util.get_color_dict()[glom], linestyle='-')
    ax4[rows[g_ind], cols[g_ind]].plot([mean_beh, mean_beh],
                                       [mean_nonbeh-err_nonbeh, mean_nonbeh+err_nonbeh],
                                       color=util.get_color_dict()[glom], linestyle='-')

    ax4[rows[g_ind], cols[g_ind]].set_axis_on()
    ax4[rows[g_ind], cols[g_ind]].spines['top'].set_visible(False)
    ax4[rows[g_ind], cols[g_ind]].spines['right'].set_visible(False)
    ax4[rows[g_ind], cols[g_ind]].set_xticks([])
    ax4[rows[g_ind], cols[g_ind]].set_yticks([])

    if p < (0.05 / len(included_gloms)):
        ax4[rows[g_ind], cols[g_ind]].annotate('*', (0, 0.55), fontsize=12)

fh4.suptitle('Mean response amplitude (dF/F)')
fh4.supxlabel('Behaving')
fh4.supylabel('Not behaving')
ax4[3, 0].set_xticks([0, 0.5])
ax4[3, 0].set_yticks([0, 0.5])

fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_summary.svg'.format(PROTOCOL_ID)), transparent=True)
fh4.savefig(os.path.join(save_directory, 'repeat_beh_{}_binary.svg'.format(PROTOCOL_ID)), transparent=True)

# %% TODO: temporal relationship between onset/offset and gain?


# %%



# %%
