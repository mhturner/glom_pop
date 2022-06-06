import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp, ttest_rel
from visanalysis.util import plot_tools
import pandas as pd
import glob
import seaborn as sns

from glom_pop import dataio, util, fictrac


PROTOCOL_ID = 'ExpandingMovingSpot'

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                          'include_in_analysis': True,
                          'diameter': 15.0,
                          }
y_min = -0.15
y_max = 0.80
eg_trials = np.arange(30, 50)

included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']

included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata=target_series_metadata,
                                                  target_groups=['aligned_response', 'behavior'])


# %%
for series in matching_series:
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('{}: {}'.format(file_name, series_number))


# %%

corr_with_running = []

response_amps = []
running_amps = []
turning_amps = []
all_behaving = []

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')

    running_amps.append(behavior_data.get('running_amp'))
    response_amps.append(response_amp)
    all_behaving.append(behavior_data.get('behaving'))

    new_beh_corr = np.array([np.corrcoef(behavior_data.get('running_amp'), response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    # Fictrac analysis:
    fps = 50  # Hz
    ft_data = pd.read_csv(glob.glob(os.path.join(ft_dir,
                                                 file_name.replace('-',''),
                                                 'series{}'.format(str(series_number).zfill(3)),
                                                 '*.dat'))[0], header=None)

    frame = ft_data.iloc[:, 0]
    zrot = ft_data.iloc[:, 7]  * 180 / np.pi * fps # rot  --> deg/sec
    zrot_filt = savgol_filter(zrot, 41, 3)

    zrot_ds = resample(zrot_filt, response_data.get('response').shape[1])
    _, turning_response_matrix = ID.getEpochResponseMatrix(zrot_ds[np.newaxis, :],
                                                           dff=False)

    turning_amp = ID.getResponseAmplitude(turning_response_matrix, metric='mean')
    turning_amps.append(turning_amp)

    # QC: check thresholding
    # fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].plot(rmse_ds)
    # ax[0].axhline(thresh, color='r')
    # ax[1].hist(rmse_ds, 100)
    # ax[1].axvline(thresh, color='r')
    # ax[0].set_title('{}: {}'.format(file_name, series_number))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        # fh0: snippet of movement and glom response traces
        fh0, ax0 = plt.subplots(3+len(included_gloms), 1, figsize=(5.5, 3.35))
        [x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
        [util.clean_axes(x) for x in ax0.ravel()]
        [x.set_ylim() for x in ax0.ravel()]

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([behavior_data.get('running_response_matrix')[:, x, :] for x in eg_trials], axis=1)
        concat_turning = np.concatenate([turning_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_behaving = np.concatenate([behavior_data.get('behavior_binary_matrix')[:, x, :] for x in eg_trials], axis=1)
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
        # ax0[0].set_ylabel('Stim', rotation=0)

        ax0[1].plot(concat_time, concat_running[0, :], color='k')
        ax0[1].set_ylim([concat_running.min(), concat_running.max()])
        ax0[1].set_ylabel('Movement', rotation=0)

        ax0[2].plot(concat_time, concat_turning[0, :], color='k')
        ax0[2].set_ylim([concat_turning.min(), concat_turning.max()])
        ax0[2].set_ylabel('Rot.', rotation=0)
        plot_tools.addScaleBars(ax0[2], dT=4, dF=200, T_value=-2, F_value=-100)

        for g_ind, glom in enumerate(included_gloms):
            ax0[3+g_ind].set_ylabel(glom, rotation=0)
            ax0[3+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
            ax0[3+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[3+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[3+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)

        # fh2: overall movement trace, with threshold and classification shading
        fh2, ax2 = plt.subplots(2, 1, figsize=(4.5, 2))

        ax2[0].set_ylabel('RMS image \ndiff. (a.u.)')

        tw_ax = ax2[0].twinx()
        tw_ax.fill_between(behavior_data['frame_times'][:len(behavior_data['binary_behavior'])],
                           behavior_data['binary_behavior'],
                           color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=0.0)
        ax2[0].axhline(behavior_data['binary_thresh'], color='r')
        ax2[0].fill_between(behavior_data['frame_times'][:len(behavior_data['rmse_smooth'])],
                         behavior_data['rmse_smooth'], y2=0,
                         color='k')
        tw_ax.set_yticks([])

        behavior_data['rmse_smooth'].shape
        behavior_data['frame_times'].shape
        zrot_filt.shape


        ax2[1].plot(behavior_data['frame_times'][:zrot_filt.shape[0]], zrot_filt, color='k')
        ax2[1].set_ylabel('Rot.\n($\degree$/sec)')
        ax2[1].set_xlabel('Time (s)')
        ax2[1].set_ylim([-250, 250])
        ax2[1].set_yticks([-200, 0, 200])
        ax2[0].set_xticks([])

corr_with_running = np.vstack(corr_with_running)  # flies x gloms
running_amps = np.vstack(running_amps)  # flies x trials
turning_amps = np.vstack(turning_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

[x.spines['top'].set_visible(False) for x in ax2]
[x.spines['right'].set_visible(False) for x in ax2]
tw_ax.spines['top'].set_visible(False)
tw_ax.spines['right'].set_visible(False)

fh0.savefig(os.path.join(save_directory, 'repeat_beh_{}_resp.svg'.format(PROTOCOL_ID)), transparent=True)
fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_running.svg'.format(PROTOCOL_ID)), transparent=True)

# %% Corr w turning
trial_gain = response_amps / np.nanmean(response_amps, axis=1)[:, np.newaxis, :]

rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]
fh, ax = plt.subplots(3, 3, figsize=(5, 4))
[x.set_xlim([-150, 150]) for x in ax.ravel()]
[x.set_ylim([0, 3.8]) for x in ax.ravel()]
[x.spines['top'].set_visible(False) for x in ax.ravel()]
[x.spines['right'].set_visible(False) for x in ax.ravel()]
[x.set_axis_off() for x in ax.ravel()]
[x.set_xticks([-100, 0, 100]) for x in ax.ravel()]
[x.set_yticks([0, 1, 2, 3]) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[rows[g_ind], cols[g_ind]].set_axis_on()
    ax[rows[g_ind], cols[g_ind]].annotate(glom, (-140, 3))
    ax[rows[g_ind], cols[g_ind]].axvline(x=0, color='k', alpha=0.5)
    ax[rows[g_ind], cols[g_ind]].axhline(y=1, color='k', alpha=0.5)
    ax[rows[g_ind], cols[g_ind]].scatter(turning_amps[:, :].ravel(), trial_gain[g_ind, :, :].T.ravel(),
                      color=util.get_color_dict()[glom], marker='.')

    if g_ind == 5:
        pass
    else:
        ax[rows[g_ind], cols[g_ind]].set_xticklabels([])
        ax[rows[g_ind], cols[g_ind]].set_yticklabels([])

fh.supylabel('Trial gain (norm.)')
fh.supxlabel('Trial average rotation ($\degree$/sec)')

fh.savefig(os.path.join(save_directory, 'repeat_beh_{}_rotation_summary.svg'.format(PROTOCOL_ID)), transparent=True)

# %% Summary plots

# For each fly: corr between trial amplitude and trial behavior amount
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 2.4))
ax2.axvline(0, color='k', alpha=0.50)
ax2.set_xlim([-0.8, 0.8])
ax2.invert_yaxis()

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], 0, nan_policy='omit')
    p_vals.append(t_result.pvalue)

    if t_result.pvalue < (0.05 / len(included_gloms)):
        ax2.annotate('*', (0.5, g_ind), fontsize=12)

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

fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_summary.svg'.format(PROTOCOL_ID)), transparent=True)
# %% (1) Response-weighted behavior
window_size = 16  # sec

def getResponseWeightedAverage(stimulus_ensemble, response_ensemble, seed=1):
        ResponseWeightedAverage = np.mean(response_ensemble * stimulus_ensemble, axis=1)
        ResponseWeightedAverage_prior = np.mean(np.mean(response_ensemble) * stimulus_ensemble, axis=1)

        np.random.seed(seed)
        ResponseWeightedAverage_random = np.mean(np.random.permutation(response_ensemble) * stimulus_ensemble, axis=1)

        results_dict = {'rwa': ResponseWeightedAverage,
                        'rwa_prior': ResponseWeightedAverage_prior,
                        'rwa_random': ResponseWeightedAverage_random}
        return results_dict

all_rwa = []
all_rwa_prior = []
all_rwa_corrected = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Get response amps and response times in raw series time
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    # Normalize within each glom
    response_amp = response_amp / np.mean(response_amp, axis=1)[:, np.newaxis]

    # Frame index of response onset, avg across all trials and gloms
    response_ind = np.argmax(np.diff(np.mean(epoch_response_matrix, axis=(0, 1))))
    # response_times: seconds into stim presentation that onset response occurs
    response_time = response_data['time_vector'][response_ind] - ID.getRunParameters('pre_time')
    stimulus_start_times = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times']
    response_times = stimulus_start_times + response_time

    behavior_rmse = behavior_data.get('rmse_smooth') / np.max(behavior_data.get('rmse_smooth'))  # Note normed behavior RMSE
    behavior_timepoints = behavior_data['frame_times'][:len(behavior_rmse)]
    stimulus_ensemble = []
    response_ensemble = []
    for pt_ind, pt in enumerate(response_times):
        # window_indices = np.where(np.logical_and(behavior_timepoints < (pt), behavior_timepoints > (pt-window_size)))[0]
        window_indices = np.where(np.logical_and(behavior_timepoints < (pt+window_size/2), behavior_timepoints > (pt-window_size/2)))[0]
        if len(window_indices) == (window_size*50):
            stimulus_ensemble.append(behavior_rmse[window_indices])
            response_ensemble.append(response_amp[:, pt_ind])

        else:
            print('Skipped epoch {}: len = {}'.format(pt_ind, len(window_indices)))


    stimulus_ensemble = np.stack(stimulus_ensemble, axis=-1)
    response_ensemble = np.stack(response_ensemble, axis=-1)

    filter_time = (1/50) * np.arange(-window_size*50, 0)
    # fh, ax = plt.subplots(len(included_gloms), 1, figsize=(2, 7))
    fly_rwa = []
    fly_rwa_prior = []
    fly_rwa_corrected = []
    for g_ind, glom in enumerate(included_gloms):
        rwa_results = getResponseWeightedAverage(stimulus_ensemble, response_ensemble[g_ind, :], seed=1)
        fly_rwa.append(rwa_results['rwa'])
        fly_rwa_prior.append(rwa_results['rwa_prior'])
        rwa_corrected = rwa_results['rwa'] - rwa_results['rwa_prior']
        # ax[g_ind].plot(filter_time, rwa_results['rwa_prior'].mean() * np.ones_like(filter_time), color=[0.5, 0.5, 0.5])
        # ax[g_ind].plot(filter_time, rwa_results['rwa'], color=util.get_color_dict()[glom])
        # ax[g_ind].plot(filter_time, rwa_results['rwa_prior'], color='k')
        fly_rwa_corrected.append(rwa_corrected)

    all_rwa.append(np.stack(fly_rwa, axis=-1))
    all_rwa_prior.append(np.stack(fly_rwa_prior, axis=-1))
    all_rwa_corrected.append(np.stack(fly_rwa_corrected, axis=-1))

all_rwa = np.stack(all_rwa, axis=-1)  # shape = time, gloms, flies
all_rwa_prior = np.stack(all_rwa_prior, axis=-1)  # shape = time, gloms, flies
all_rwa_corrected = np.stack(all_rwa_corrected, axis=-1)  # shape = time, gloms, flies

# %% Plot response-weighted behavior...

filter_time = (1/50) * np.arange(-window_size/2*50, window_size/2*50)

fh1, ax1 = plt.subplots(len(included_gloms), 1, figsize=(1.5, 7))
[x.set_ylim([-0.030, 0.01]) for x in ax1.ravel()]
# [x.set_xlim([-5, 0]) for x in ax1.ravel()]
[x.set_yticks([]) for x in ax1[:-1]]
[x.set_xticks([]) for x in ax1[:-1]]
[x.spines['top'].set_visible(False) for x in ax1.ravel()]
[x.spines['right'].set_visible(False) for x in ax1.ravel()]
for g_ind, glom in enumerate(included_gloms):
    corrected_mean = np.nanmean(all_rwa_corrected[:, g_ind, :], axis=-1)
    corrected_sem = np.nanstd(all_rwa_corrected[:, g_ind, :], axis=-1) / all_rwa.shape[-1]
    ax1[g_ind].axhline(y=0, color=[0.5, 0.5, 0.5])
    ax1[g_ind].fill_between(filter_time,
                           corrected_mean-corrected_sem,
                           corrected_mean+corrected_sem,
                           color=util.get_color_dict()[glom], linewidth=0)
    ax1[g_ind].plot(filter_time, corrected_mean, color=util.get_color_dict()[glom], linewidth=2)

ax1[g_ind].set_xlabel('Time to response onset (s)')
fh1.supylabel('Response-weighted behavior (a.u.)')

# %% (2) Response gain as a function of time since behavior onset/offset

from scipy.signal import medfilt
from skimage import filters

eg_glom_ind = 0
window_size = 3  # sec

minimum_bout_duration = 0.5  # sec
minimum_before_duration = 1.0  # sec
minimum_after_duration = 1.0  # sec


# series_number=1
# file_path = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/datafiles/2022-04-12.hdf5'
# file_name = os.path.split(file_path)[-1]
# ID = imaging_data.ImagingDataObject(file_path,
#                                     series_number,
#                                     quiet=True)

onset_dt = []
onset_amp = []

offset_dt = []
offset_amp = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)


    # Get response amps and response times in raw series time
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    # Normalize within each glom
    response_amp = response_amp / np.mean(response_amp, axis=1)[:, np.newaxis]

    # Frame index of peak response, avg across all trials and gloms
    onset_ind = np.argmax(np.diff(np.mean(epoch_response_matrix, axis=(0, 1))))
    # onset_time: seconds into stim presentation that onset response occurs
    onset_time = response_data['time_vector'][onset_ind] - ID.getRunParameters('pre_time')
    stimulus_start_times = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times']
    onset_tims = stimulus_start_times + onset_time

    # Get behavior trace, smooth and binarize to find "bouts"
    """
    """
    behavior_timepoints = behavior_data['frame_times'][:len(behavior_data.get('rmse_smooth'))]
    thresh = filters.threshold_li(behavior_data.get('rmse_smooth'))
    beh_binary = behavior_data.get('rmse_smooth') > thresh

    dbinary = beh_binary[1:].astype(int) - beh_binary[:-1].astype(int)
    behavior_onsets = np.where(dbinary == 1)[0]
    behavior_offsets = np.where(dbinary == -1)[0]

    onset_times = behavior_timepoints[behavior_onsets]
    offset_times = behavior_timepoints[behavior_offsets]

    if onset_times[0] > offset_times[0]:  # animal starts off behaving. First transition is 1->0. Chop this out
        offset_times = offset_times[1:]

    assert onset_times[0] < offset_times[0]
    assert len(onset_times) == len(offset_times)

    # fh, ax = plt.subplots(2, 1, figsize=(4, 4))
    # [x.axhline(y=np.nanmean(response_amp[eg_glom_ind, :])) for x in ax]
    num_bouts = len(onset_times)
    for bout in range(1, num_bouts-1):
        bout_start = onset_times[bout]
        bout_end = offset_times[bout]
        bout_duration = bout_end - bout_start  # sec
        # duration (sec) of pre/post-bout quiet periods
        before_duration = bout_start - offset_times[bout-1]
        after_duration = offset_times[bout+1] - bout_end

        if bout_duration >= minimum_bout_duration:
            # Get response amplitudes around the bout start
            if before_duration >= minimum_before_duration:
                inds = np.where(np.logical_and(onset_times > (bout_start-window_size/2), onset_times < (bout_start+window_size/2)))[0]
                if len(inds) > 0:
                    new_dt = onset_times[inds] - bout_start
                    new_amp = response_amp[eg_glom_ind, inds]
                    # ax[0].plot(new_dt, new_amp, linestyle='None', marker='.', color='k')
                    onset_dt.append(new_dt)
                    onset_amp.append(new_amp)

            # Get response amplitudes around the bout end
            if after_duration >= minimum_after_duration:
                inds = np.where(np.logical_and(onset_times > (bout_end-window_size/2), onset_times < (bout_end+window_size/2)))[0]
                if len(inds) > 0:
                    new_dt = onset_times[inds] - bout_end
                    new_amp = response_amp[eg_glom_ind, inds]
                    # ax[1].plot(new_dt, new_amp, linestyle='None', marker='.', color='k')
                    offset_dt.append(new_dt)
                    offset_amp.append(new_amp)

onset_dt = np.hstack(onset_dt)
onset_amp = np.hstack(onset_amp)

offset_dt = np.hstack(offset_dt)
offset_amp = np.hstack(offset_amp)


# %%
fh, ax = plt.subplots(2, 1, figsize=(4, 4))
[x.axhline(y=np.nanmean(response_amp[eg_glom_ind, :])) for x in ax]
ax[0].plot(onset_dt, onset_amp, linestyle='None', marker='.', color='k')
ax[1].plot(offset_dt, offset_amp, linestyle='None', marker='.', color='k')

# %%



for on_ind, onset_time in enumerate(onset_times):
    if behavior_durations[on_ind] > thresh:
        inds = np.where(np.logical_and(onset_times > onset_time, onset_times < (onset_time+window_size)))[0]
        if len(inds) > 0:
            new_dt = onset_times[inds] - onset_time
            new_amp = response_amp[eg_glom_ind, inds]


for off_ind, offset_time in enumerate(offset_times):
    previous_bout_duration = behavior_durations[off_ind-1]


# %%
#





# %%
