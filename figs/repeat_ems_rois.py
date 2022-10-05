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
                                                                       'indicator_1': 'GCaMP6f'},
                                                  target_series_metadata={'protocol_ID': 'ExpandingMovingSpot'},
                                                  target_roi_series=['dend_stalk']
                                                                          )
matching_series
# %%
eg_s_ind = 2
target_roi = 'dend_stalk'
y_min = -0.1
y_max = 0.3
# eg_trials = np.arange(20, 40)
eg_trials = np.arange(0, 100)

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
    epoch_response_matrix = np.mean(ID.getRoiResponses(target_roi)['epoch_response'], axis=0) # avg across all rois
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    response_amps.append(response_amp)

    # # Fictrac data:
    ft_data_path = glob.glob(os.path.join(ft_dir,
                                          file_name.replace('-', ''),
                                          'series{}'.format(str(series_number).zfill(3)),
                                          '*.dat'))[0]
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             process_behavior=True, exclude_thresh=300, show_qc=False)
    walking_amps.append(behavior_data.get('walking_amp'))

    new_beh_corr = np.corrcoef(behavior_data.get('walking_amp'), response_amp)[0, 1]

    corr_with_running.append(new_beh_corr)

    # if s_ind == eg_s_ind:
    if True:
        behaving_trial_matrix = np.zeros_like(behavior_data.get('walking_response_matrix'))
        behaving_trial_matrix[behavior_data.get('is_behaving'), :] = 1

        # fh0: snippet of movement and glom response traces
        concat_response = np.concatenate([epoch_response_matrix[x, :] for x in eg_trials], axis=0)
        concat_walking = np.concatenate([behavior_data.get('walking_response_matrix')[:, x, :] for x in eg_trials], axis=1)
        concat_behaving = np.concatenate([behaving_trial_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_walking.shape[1]) * ID.getAcquisitionMetadata('sample_period')

        # Red triangles when stim hits center of screen (middle of trial)
        dt = np.diff(concat_time)[0]  # sec
        trial_len = epoch_response_matrix.shape[1]
        concat_len = len(concat_time)
        trial_time = dt * np.linspace(trial_len/2,
                                     concat_len-trial_len/2,
                                     len(eg_trials))
        y_val = 0.5

        fh0, ax0 = plt.subplots(3, 1, figsize=(20, 2))
        [util.clean_axes(x) for x in ax0.ravel()]
        ax0[0].plot(trial_time,
                    y_val * np.ones(len(eg_trials)),
                    'rv', markersize=4)
        ax0[0].set_ylim([0.25, 0.75])
        # ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')

        ax0[1].plot(concat_time, concat_walking[0, :], color='k')
        # ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
        ax0[1].set_ylabel('Walking', rotation=0)

        ax0[2].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
        ax0[2].plot(concat_time, concat_response)
        ax0[2].set_ylim([concat_response.min(), concat_response.max()])
        plot_tools.addScaleBars(ax0[2], dT=4, dF=0.1, T_value=0, F_value=-0.1)



corr_with_running = np.vstack(corr_with_running)  # flies x gloms
walking_amps = np.vstack(walking_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

# fh0.savefig(os.path.join(save_directory, 'LC11_repeat_beh_dend_resp.svg'), transparent=True)

print(corr_with_running)
# corr_with_running.mean(axis=0)
# TODO check out timing on S=2, some issue with frame timing maybe?




# %%
