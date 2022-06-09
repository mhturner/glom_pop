import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
from visanalysis.util import plot_tools
import pandas as pd
import glob
from skimage import filters

from glom_pop import dataio, util


PROTOCOL_ID = 'ExpandingMovingSpot'
# PROTOCOL_ID = 'LoomingSpot'

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

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

    included_vals = dataio.get_glom_vals_from_names(included_gloms)

    matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                      target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                           'indicator_1': 'Syt1GCaMP6f',
                                                                           'indicator_2': 'TdTomato'},
                                                      target_series_metadata=target_series_metadata,
                                                      target_groups=['aligned_response', 'behavior'])
elif PROTOCOL_ID == 'LoomingSpot':
    eg_series = ('2022-04-04', 2)
    target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                              'include_in_analysis': True,
                              'rv_ratio': 100.0,
                              'center': [0, 0],
                              }
    y_min = -0.05
    y_max = 0.35
    eg_trials = np.arange(0, 20)
    included_gloms = ['LC6', 'LC26', 'LPLC2', 'LC4', 'LPLC1', 'LC9', 'LC17', 'LC12']

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
walking_amps = []
all_behaving = []

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
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    response_amps.append(response_amp)

    # # Fictrac data:
    ft_data_path = glob.glob(os.path.join(ft_dir,
                                          file_name.replace('-', ''),
                                          'series{}'.format(str(series_number).zfill(3)),
                                          '*.dat'))[0]
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             response_len = response_data.get('response').shape[1],
                                             process_behavior=True, fps=50)
    walking_amps.append(behavior_data.get('walking_amp'))

    new_beh_corr = np.array([np.corrcoef(behavior_data.get('walking_amp'), response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    # # QC: check thresholding
    # fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    # behavior_data.get('walking_mag_ds')
    # ax[0].plot(behavior_data.get('walking_mag_ds'))
    # ax[0].axhline(behavior_data.get('thresh'), color='r')
    # ax[1].hist(behavior_data.get('walking_mag_ds'), 100)
    # ax[1].axvline(behavior_data.get('thresh'), color='r')
    # ax[0].set_title('{}: {}'.format(file_name, series_number))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        trial_time = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times'] - ID.getRunParameters('pre_time')
        f_interp_behaving = interp1d(trial_time, behavior_data.get('is_behaving')[0, :], kind='previous', bounds_error=False, fill_value=np.nan)

        # fh0: snippet of movement and glom response traces
        fh0, ax0 = plt.subplots(2+len(included_gloms), 1, figsize=(5.5, 3.35))
        [x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
        [util.clean_axes(x) for x in ax0.ravel()]
        [x.set_ylim() for x in ax0.ravel()]

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_walking = np.concatenate([behavior_data.get('walking_response_matrix')[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_walking.shape[1]) * ID.getAcquisitionMetadata('sample_period')
        concat_behaving = f_interp_behaving(trial_time[eg_trials][0]+concat_time)

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

        ax0[1].plot(concat_time, concat_walking[0, :], color='k')
        ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
        ax0[1].set_ylabel('Walking', rotation=0)

        for g_ind, glom in enumerate(included_gloms):
            ax0[2+g_ind].set_ylabel(glom, rotation=0)
            ax0[2+g_ind].fill_between(concat_time, concat_behaving, color='k', alpha=0.25, linewidth=0)
            ax0[2+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[2+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[2+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)

        # fh2: overall movement trace, with threshold and classification shading
        time_ds = ID.getAcquisitionMetadata('sample_period') * np.arange(response_data.get('response').shape[1])
        fh2, ax2 = plt.subplots(1, 1, figsize=(4.5, 1.5))
        ax2.set_ylabel('Walking amp.')
        tw_ax = ax2.twinx()
        tw_ax.fill_between(time_ds,
                           f_interp_behaving(time_ds),
                           color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=0.0)
        ax2.axhline(behavior_data.get('thresh'), color='r')
        ax2.plot(time_ds, behavior_data.get('walking_mag_ds'),
                         color='k')
        tw_ax.set_yticks([])

        ax2.set_xlabel('Time (s)')

corr_with_running = np.vstack(corr_with_running)  # flies x gloms
walking_amps = np.vstack(walking_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tw_ax.spines['top'].set_visible(False)
tw_ax.spines['right'].set_visible(False)

fh0.savefig(os.path.join(save_directory, 'repeat_beh_{}_resp.svg'.format(PROTOCOL_ID)), transparent=True)
fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_running.svg'.format(PROTOCOL_ID)), transparent=True)

# %% Summary plots

# For each fly: corr between trial amplitude and trial behavior amount
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 2.6))
ax2.axvline(0, color='k', alpha=0.50)
ax2.set_xlim([-0.8, 0.8])
ax2.invert_yaxis()

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], popmean=0, nan_policy='omit')
    p_vals.append(t_result.pvalue)


    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax2.plot(corr_with_running[:, g_ind], g_ind * np.ones(corr_with_running.shape[0]),
             marker='.', color=util.get_color_dict()[glom], linestyle='none', alpha=0.5)

    ax2.plot(y_mean, g_ind,
             marker='o', color=util.get_color_dict()[glom])

    ax2.plot([y_mean-y_err, y_mean+y_err], [g_ind, g_ind],
             color=util.get_color_dict()[glom])

# Multiple comparisons test. Step down bonferroni
h, p_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='holm')
for g_ind, glom in enumerate(included_gloms):
    if h[g_ind]:
        ax2.annotate('*', (0.5, g_ind), fontsize=12)

ax2.set_yticks([])
ax2.set_xlabel('Corr. with behavior (r)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

fh2.savefig(os.path.join(save_directory, 'repeat_beh_{}_summary.svg'.format(PROTOCOL_ID)), transparent=True)


# %% OLD STUFF

# Corr w turning
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
