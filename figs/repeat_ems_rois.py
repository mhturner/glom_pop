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
y_min = -0.15
y_max = 0.8
eg_trials = np.arange(20, 40)

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

    if s_ind == eg_s_ind:
    # if True:
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

        fh0, ax0 = plt.subplots(3, 1, figsize=(5.5, 2))
        [util.clean_axes(x) for x in ax0.ravel()]
        ax0[0].plot(trial_time,
                    y_val * np.ones(len(eg_trials)),
                    'rv', markersize=4)
        ax0[0].set_ylim([0.25, 0.75])

        ax0[1].plot(concat_time, concat_walking[0, :], color='k')
        ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
        ax0[1].set_ylabel('Walking', rotation=0)

        ax0[2].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
        ax0[2].plot(concat_time, concat_response, color=util.get_color_dict()['LC11'])
        ax0[2].set_ylabel('LC11', rotation=0)
        plot_tools.addScaleBars(ax0[2], dT=4, dF=0.25, T_value=0, F_value=-0.1)



corr_with_running = np.vstack(corr_with_running)  # flies x gloms
walking_amps = np.vstack(walking_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

fh0.savefig(os.path.join(save_directory, 'LC11_repeat_beh_dend_resp.svg'), transparent=True)


# %% Roi map
fh1 = ID.generateRoiMap('dend_stalk', z=5, return_fighandle=True)
ax = fh1.axes[0]
ax.set_ylim([150, 25])
fh1.set_size_inches(1.2, 1.2)
fh1.savefig(os.path.join(save_directory, 'LC11_repeat_beh_dend_roimap.svg'), transparent=True)

# %%

fh2, ax2 = plt.subplots(1, 1, figsize=(2, 1))
ax2.axvline(0, color='k', alpha=0.50)
ax2.set_xlim([-0.8, 0.8])
ax2.invert_yaxis()

t_result = ttest_1samp(corr_with_running, popmean=0, nan_policy='omit')

y_mean = np.nanmean(corr_with_running)
y_err = np.nanstd(corr_with_running) / np.sqrt(corr_with_running.shape[0])
ax2.plot(corr_with_running, np.ones(corr_with_running.shape[0]),
         marker='.', color=util.get_color_dict()['LC11'], linestyle='none', alpha=0.5)

ax2.plot(y_mean, 1,
         marker='o', color=util.get_color_dict()['LC11'])

ax2.plot([y_mean-y_err, y_mean+y_err], [1, 1],
         color=util.get_color_dict()['LC11'])
print('p = {:.4f}'.format(t_result.pvalue[0]))
if t_result.pvalue < 0.05:
    ax2.annotate('*', (0.5, 1), fontsize=12)

ax2.set_yticks([])
ax2.set_xlabel(r'Corr. with behavior ($\rho$)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

fh2.savefig(os.path.join(save_directory, 'LC11_repeat_beh_dend_summary.svg'), transparent=True)


# %%
roi_ind = 1
roi_data = ID.getRoiResponses(target_roi)

fh3, ax3 = plt.subplots(1, 1, figsize=(1.5, 1.5))
epoch_response_matrix = roi_data['epoch_response'][roi_ind, :, :]
epoch_response_matrix.shape
time_vec = roi_data['time_vector']
trial_avg = np.mean(epoch_response_matrix, axis=0)
trial_err = np.std(epoch_response_matrix, axis=0) / np.sqrt(epoch_response_matrix.shape[0])
ax3.axhline(y=0, color=[0.5, 0.5, 0.5], alpha=0.5)
ax3.fill_between(time_vec, trial_avg-trial_err, trial_avg+trial_err, color='k', alpha=0.5)
ax3.plot(time_vec, trial_avg, color='k', linewidth=2)

util.clean_axes(ax3)
plot_tools.addScaleBars(ax3, dT=2, dF=0.25, T_value=-0.2, F_value=-0.1)

fh3.savefig(os.path.join(save_directory, 'LC11_repeat_trial_avg_trace_.svg'), transparent=True)



# %%
