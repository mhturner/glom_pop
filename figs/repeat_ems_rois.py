import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp, ttest_rel
from scipy.interpolate import interp1d
from visanalysis.util import plot_tools
# from skimage import filters
import pandas as pd
import glob

from glom_pop import dataio, util

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'LC11',
                                                                       'indicator_1': 'GCaMP6f',
                                                                       },
                                                  target_series_metadata={'protocol_ID': 'ExpandingMovingSpot',
                                                                          'diameter': 15.0})


# %%
y_min = -0.1
y_max = 0.3
eg_trials = np.arange(40, 60)

response_amps = []
walking_amps = []
corr_with_running = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    glom_erm = ID.getRoiResponses('glom')['epoch_response']
    dend_erm = np.mean(ID.getRoiResponses('dend')['epoch_response'], axis=0)
    stalk_erm = np.mean(ID.getRoiResponses('dend_stalk')['epoch_response'], axis=0)
    all_erm = np.stack([glom_erm[0, ...], dend_erm, stalk_erm], axis=0)

    roi_data = ID.getRoiResponses('glom')
    response_amp = ID.getResponseAmplitude(all_erm, metric='max')
    response_amps.append(response_amp)

    # # Fictrac data:
    response_len = roi_data.get('roi_response')[0].shape[1]
    ft_data_path = glob.glob(os.path.join(ft_dir,
                                          file_name.replace('-', ''),
                                          'series{}'.format(str(series_number).zfill(3)),
                                          '*.dat'))[0]
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             response_len,
                                             process_behavior=True, fps=50)
    walking_amps.append(behavior_data.get('walking_amp'))

    new_beh_corr = np.array([np.corrcoef(behavior_data.get('walking_amp'), response_amp[x, :])[0, 1] for x in range(response_amp.shape[0])])
    corr_with_running.append(new_beh_corr)

    # # QC: check thresholding
    # fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    # behavior_data.get('walking_mag_ds')
    # ax[0].plot(behavior_data.get('walking_mag_ds'))
    # ax[0].axhline(behavior_data.get('thresh'), color='r')
    # ax[1].hist(behavior_data.get('walking_mag_ds'), 100)
    # ax[1].axvline(behavior_data.get('thresh'), color='r')
    # ax[0].set_title('{}: {}'.format(file_name, series_number))

    # if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    if s_ind == 0:
        trial_time = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times'] - ID.getRunParameters('pre_time')
        f_interp_behaving = interp1d(trial_time, behavior_data.get('is_behaving')[0, :], kind='previous', bounds_error=False, fill_value=np.nan)

        # fh0: snippet of movement and glom response traces
        fh0, ax0 = plt.subplots(2+response_amp.shape[0], 1, figsize=(5.5, 3.35))
        # fh0, ax0 = plt.subplots(2+response_amp.shape[0], 1, figsize=(18, 2))
        [x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
        [util.clean_axes(x) for x in ax0.ravel()]
        [x.set_ylim() for x in ax0.ravel()]
        concat_response = np.concatenate([all_erm[:, x, :] for x in eg_trials], axis=1)
        concat_walking = np.concatenate([behavior_data.get('walking_response_matrix')[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_walking.shape[1]) * ID.getAcquisitionMetadata('sample_period')
        concat_behaving = f_interp_behaving(trial_time[eg_trials][0]+concat_time)

        # Red triangles when stim hits center of screen (middle of trial)
        dt = np.diff(concat_time)[0]  # sec
        trial_len = all_erm.shape[2]
        concat_len = len(concat_time)
        y_val = 0.5
        ax0[0].plot(dt * np.linspace(trial_len/2,
                                     concat_len-trial_len/2,
                                     len(eg_trials)),
                    y_val * np.ones(len(eg_trials)),
                    'rv', markersize=4)
        ax0[0].set_ylim([0.25, 0.75])
        # ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')

        ax0[1].plot(concat_time, concat_walking[0, :], color='k')
        ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
        ax0[1].set_ylabel('Walking', rotation=0)

        labels = ['glom', 'dend', 'stalk']
        for r_ind in range(concat_response.shape[0]):
            ax0[2+r_ind].set_ylabel(labels[r_ind], rotation=0)
            ax0[2+r_ind].fill_between(concat_time, concat_behaving, color='k', alpha=0.25, linewidth=0)
            ax0[2+r_ind].plot(concat_time, concat_response[r_ind, :])
            ax0[2+r_ind].set_ylim([concat_response[r_ind, :].min(), concat_response[r_ind, :].max()])
            if r_ind == 0:
                plot_tools.addScaleBars(ax0[2+r_ind], dT=4, dF=0.1, T_value=-1, F_value=-0.1)
            else:
                plot_tools.addScaleBars(ax0[2+r_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)


corr_with_running = np.vstack(corr_with_running)  # flies x gloms
walking_amps = np.vstack(walking_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# tw_ax.spines['top'].set_visible(False)
# tw_ax.spines['right'].set_visible(False)

fh0.savefig(os.path.join(save_directory, 'LC11_repeat_beh_{}_resp.svg'.format(PROTOCOL_ID)), transparent=True)
fh2.savefig(os.path.join(save_directory, 'LC11_repeat_beh_{}_running.svg'.format(PROTOCOL_ID)), transparent=True)

print(corr_with_running)
corr_with_running.mean(axis=0)

# %%
