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

from glom_pop import dataio, util


PROTOCOL_ID = 'ExpandingMovingSpot'


sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')

eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                          'include_in_analysis': True,
                          'diameter': 15.0,
                          }
y_min = -0.15
y_max = 0.80

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata=target_series_metadata,
                                                  target_groups=['behavior'],
                                                  target_roi_series=['lo_prox', 'lo_dist'])


# %%
eg_trials = np.arange(30, 50)


# eg fly figs...
# fh0: snippet of movement and glom response traces
fh0, ax0 = plt.subplots(4, 1, figsize=(5.5, 3))
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
    epoch_response_matrix_prox = ID.getRoiResponses('lo_prox').get('epoch_response')
    epoch_response_matrix_dist = ID.getRoiResponses('lo_dist').get('epoch_response')
    epoch_response_matrix = np.vstack([epoch_response_matrix_prox, epoch_response_matrix_dist])
    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')

    running_amps.append(behavior_data.get('running_amp'))
    response_amps.append(response_amp)
    all_behaving.append(behavior_data.get('behaving'))

    new_beh_corr = np.corrcoef(behavior_data.get('running_amp'), response_amp[0, :])[0, 1]
    corr_with_running.append(new_beh_corr)


    # if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    if True:
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

        for roi_ind, roi in enumerate(['prox', 'dist']):

            ax0[roi_ind+2].set_ylabel(roi, rotation=0)
            ax0[roi_ind+2].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
            ax0[roi_ind+2].plot(concat_time, concat_response[roi_ind, :], color='k')
            ax0[roi_ind+2].set_ylim([concat_response.min(), concat_response.max()])
        plot_tools.addScaleBars(ax0[2], dT=4, dF=0.25, T_value=-1, F_value=-0.1)

        # Image of fly on ball:
        ax1.imshow(behavior_data['frame'], cmap='Greys_r')

        # Fly movement traj with thresh and binary shading
        tw_ax = ax2.twinx()
        tw_ax.fill_between(behavior_data['frame_times'][:len(behavior_data['binary_behavior'])],
                           behavior_data['binary_behavior'],
                           color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=0.0)
        ax2.axhline(behavior_data['binary_thresh'], color='r')
        ax2.fill_between(behavior_data['frame_times'][:len(behavior_data['rmse_smooth'])],
                         behavior_data['rmse_smooth'], y2=0,
                         color='k')
        tw_ax.set_yticks([])

corr_with_running = np.vstack(corr_with_running)  # flies x gloms
running_amps = np.vstack(running_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tw_ax.spines['top'].set_visible(False)
tw_ax.spines['right'].set_visible(False)

# %%




# %%
