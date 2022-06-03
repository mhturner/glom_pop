import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp, ttest_rel
from visanalysis.util import plot_tools
# from skimage import filters
import pandas as pd
import glob

from glom_pop import dataio, util, fictrac


PROTOCOL_ID = 'ExpandingMovingSpot'
# PROTOCOL_ID = 'LoomingSpot'


sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')

if PROTOCOL_ID == 'ExpandingMovingSpot':
    eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
    target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                              'include_in_analysis': True,
                              'diameter': 15.0,
                              }
    y_min = -0.15
    y_max = 0.80
    eg_trials = np.arange(30, 50)

    included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']


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

# %% Fictrac data analyze
series_number = 1
file_path = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/datafiles/2022-04-12.hdf5'
file_name = os.path.split(series['file_name'])[-1]
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# Get behavior data
behavior_data = dataio.load_behavior(ID, process_behavior=True)

# FT data:
ft_dir = '/Users/mhturner/Desktop/'
dir = os.path.join(ft_dir, 'series001')
filename = os.path.split(glob.glob(os.path.join(dir, '*.dat'))[0])[-1]

ft_data = pd.read_csv(os.path.join(dir, filename), header=None)
sphere_radius = 4.5e-3 # in m
fps = 50  # hz

frame = ft_data.iloc[:, 0]
xrot = ft_data.iloc[:, 5]
yrot = ft_data.iloc[:, 6] * sphere_radius * fps * 1000 # fwd --> in mm/sec
zrot = ft_data.iloc[:, 7]  * 180 / np.pi * fps # rot  --> deg/sec

heading = ft_data.iloc[:, 16]
direction = ft_data.iloc[:, 16] + ft_data.iloc[:, 17]

speed = ft_data.iloc[:, 18]

x_loc = ft_data.iloc[:, 14]
y_loc = ft_data.iloc[:, 15]

plt.plot(x_loc, y_loc)

# %%
xrot_filt = savgol_filter(xrot, 41, 3)
yrot_filt = savgol_filter(yrot, 41, 3)
zrot_filt = savgol_filter(zrot, 41, 3)

timestamps = 1/50 * np.arange(0, len(frame))
fh, ax = plt.subplots(3, 1, figsize=(16, 8))
ax[0].plot(timestamps, yrot_filt, 'k')
ax[1].plot(timestamps, zrot_filt, 'b')


ax[2].plot(timestamps[:-1], behavior_data.get('rmse'), 'r')

[x.set_xlim([100, 200]) for x in ax]

# plt.hist(zrot_filt, 100)

# %%

for series in matching_series:
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('{}: {}'.format(file_name, series_number))


# %%
# eg fly figs...
# fh0: snippet of movement and glom response traces
fh0, ax0 = plt.subplots(2+len(included_gloms), 1, figsize=(5.5, 3))
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

    # QC: check thresholding
    # fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].plot(rmse_ds)
    # ax[0].axhline(thresh, color='r')
    # ax[1].hist(rmse_ds, 100)
    # ax[1].axvline(thresh, color='r')
    # ax[0].set_title('{}: {}'.format(file_name, series_number))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        # # FT data:
        # ft_dir = '/Users/mhturner/Desktop/'
        # ft_data = pd.read_csv(glob.glob(os.path.join(ft_dir, 'series001', '*.dat'))[0], header=None)
        #
        # frame = ft_data.iloc[:, 0]
        # heading = ft_data.iloc[:, 16]
        # direction = ft_data.iloc[:, 16] + ft_data.iloc[:, 17]
        # speed = ft_data.iloc[:, 18]
        #
        # direction = np.mod(direction, 2*np.pi)
        # fh, ax = plt.subplots(2, 2, figsize=(16, 8))
        # ax[0, 0].plot(frame, speed, 'b')
        # ax[0, 1].plot(frame, direction, 'k')
        #
        # ax[1, 0].plot(behavior_data.get('rmse'), 'b')

        #
        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([behavior_data.get('running_response_matrix')[:, x, :] for x in eg_trials], axis=1)
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
        ax0[0].set_ylabel('Stim', rotation=0)

        ax0[1].plot(concat_time, concat_running[0, :], color='k')
        ax0[1].set_ylim([concat_running.min(), concat_running.max()])
        ax0[1].set_ylabel('Movement', rotation=0)

        for g_ind, glom in enumerate(included_gloms):
            ax0[2+g_ind].set_ylabel(glom, rotation=0)
            ax0[2+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
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
                           color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=0.0)
        ax2.axhline(behavior_data['binary_thresh'], color='r')
        ax2.plot(behavior_data['frame_times'][:len(behavior_data['rmse_smooth'])],
                 behavior_data['rmse_smooth'],
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
        ax4[rows[g_ind], cols[g_ind]].annotate('*', (0, 0.45), fontsize=12)

fh4.suptitle('Mean response amplitude (dF/F)')
fh4.supxlabel('Behaving')
fh4.supylabel('Not behaving')
ax4[3, 0].set_xticks([0, 0.5])
ax4[3, 0].set_yticks([0, 0.5])

fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_summary.svg'.format(PROTOCOL_ID)), transparent=True)
fh4.savefig(os.path.join(save_directory, 'repeat_beh_{}_binary.svg'.format(PROTOCOL_ID)), transparent=True)
# %% Temporal relationship between behavior and response gain modulation...
# %% (1) Response-weighted behavior
window_size = 8  # sec

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

    # Frame index of peak response, avg across all trials and gloms
    peak_ind = np.argmax(np.mean(epoch_response_matrix, axis=(0, 1)), axis=-1)  # shape = ngloms
    # peak_time: seconds into stim presentation that peak response occurs
    peak_time = response_data['time_vector'][peak_ind] - ID.getRunParameters('pre_time')
    stimulus_start_times = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times']
    peak_times = stimulus_start_times + peak_time

    behavior_rmse = behavior_data.get('rmse_smooth') / np.max(behavior_data.get('rmse_smooth'))  # Note normed behavior RMSE
    behavior_timepoints = behavior_data['frame_times'][:len(behavior_rmse)]
    stimulus_ensemble = []
    response_ensemble = []
    for pt_ind, pt in enumerate(peak_times):
        window_indices = np.where(np.logical_and(behavior_timepoints < (pt), behavior_timepoints > (pt-window_size)))[0]
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

fh1, ax1 = plt.subplots(len(included_gloms), 1, figsize=(1.5, 7))
[x.set_ylim([-0.030, 0.01]) for x in ax1.ravel()]
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

ax1[g_ind].set_xlabel('Time to response peak (s)')
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
    peak_ind = np.argmax(np.mean(epoch_response_matrix, axis=(0, 1)), axis=-1)  # shape = ngloms
    # peak_time: seconds into stim presentation that peak response occurs
    peak_time = response_data['time_vector'][peak_ind] - ID.getRunParameters('pre_time')
    stimulus_start_times = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times']
    peak_times = stimulus_start_times + peak_time

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
                inds = np.where(np.logical_and(peak_times > (bout_start-window_size/2), peak_times < (bout_start+window_size/2)))[0]
                if len(inds) > 0:
                    new_dt = peak_times[inds] - bout_start
                    new_amp = response_amp[eg_glom_ind, inds]
                    # ax[0].plot(new_dt, new_amp, linestyle='None', marker='.', color='k')
                    onset_dt.append(new_dt)
                    onset_amp.append(new_amp)

            # Get response amplitudes around the bout end
            if after_duration >= minimum_after_duration:
                inds = np.where(np.logical_and(peak_times > (bout_end-window_size/2), peak_times < (bout_end+window_size/2)))[0]
                if len(inds) > 0:
                    new_dt = peak_times[inds] - bout_end
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
        inds = np.where(np.logical_and(peak_times > onset_time, peak_times < (onset_time+window_size)))[0]
        if len(inds) > 0:
            new_dt = peak_times[inds] - onset_time
            new_amp = response_amp[eg_glom_ind, inds]


for off_ind, offset_time in enumerate(offset_times):
    previous_bout_duration = behavior_durations[off_ind-1]


# %%
#
# window_radius = 3  # sec
# eg_glom_ind = 0
# onset_times = behavior_timepoints[behavior_onsets]
# offset_times = behavior_timepoints[behavior_offsets]
#
# onset_dt = []
# onset_amp = []
# for on_ind, onset_time in enumerate(onset_times):
#     # find nearest response before onset
#     tmp = np.where((peak_times < onset_time))[0]
#     if len(tmp) > 0:
#         before_ind = tmp[-1]
#         dt = peak_times[before_ind] - onset_time
#         resp = response_amp[eg_glom_ind, before_ind]
#         onset_dt.append(dt)
#         onset_amp.append(resp)
#
#     # find nearest response after onset
#     tmp = np.where((peak_times > onset_time))[0]
#     if len(tmp) > 0:
#         after_ind = tmp[0]
#         dt = peak_times[after_ind] - onset_time
#         resp = response_amp[eg_glom_ind, after_ind]
#         onset_dt.append(dt)
#         onset_amp.append(resp)
#
# onset_dt = np.hstack(onset_dt)
# onset_amp = np.hstack(onset_amp)
#
# # %%
# fh, ax = plt.subplots(2, 1, figsize=(4, 4))
# ax[0].plot(onset_dt, onset_amp, 'k.')
# ax[0].axhline(y=np.mean(all_amps), color='k', alpha=0.5)
#
#
# # %%
# np.logical_and(peak_times > (onset_time-window_radius), peak_times < (onset_time+window_radius))
#
# offset_dt = []
# offset_amp = []
# for off_ind, offset_time in enumerate(offset_times):
#     response_indices = np.logical_and(peak_times > (offset_time-window_radius), peak_times < (offset_time+window_radius))
#     window_amplitudes = response_amp[eg_glom_ind, response_indices]
#     window_dts = peak_times[response_indices] - offset_time
#     offset_dt.append(window_dts)
#     offset_amp.append(window_amplitudes)
# offset_dt = np.hstack(offset_dt)
# offset_amp = np.hstack(offset_amp)
# # %%
# fh, ax = plt.subplots(2, 1, figsize=(4, 4))
# ax[0].plot(onset_dt, onset_amp, 'k.')
# ax[0].axhline(y=np.mean(all_amps), color='k', alpha=0.5)
#
#
# ax[1].plot(offset_dt, offset_amp, 'k.')
# ax[1].axhline(y=np.mean(offset_amp), color='k', alpha=0.5)


# %%

onset_dt[onset_sort_inds]
onset_amp[onset_sort_inds]
onset_dt.max()

# %%


# %%






# %%
