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


PROTOCOL_ID = 'LoomingSpot'

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

eg_series = ('2022-04-04', 2)
target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                          'include_in_analysis': True,
                          'rv_ratio': 100.0,
                          'center': [0, 0],
                          }
y_min = -0.05
y_max = 0.35
eg_trials = np.arange(0, 20)
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
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
        # # Fictrac analysis:
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
        plot_tools.addScaleBars(ax0[2], dT=4, dF=200, T_value=-2, F_value=-50)

        for g_ind, glom in enumerate(included_gloms):
            ax0[3+g_ind].set_ylabel(glom, rotation=0)
            ax0[3+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
            ax0[3+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[3+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[3+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)


corr_with_running = np.vstack(corr_with_running)  # flies x gloms
running_amps = np.vstack(running_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tw_ax.spines['top'].set_visible(False)
tw_ax.spines['right'].set_visible(False)

fh0.savefig(os.path.join(save_directory, 'repeat_beh_{}_resp.svg'.format(PROTOCOL_ID)), transparent=True)

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
