import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample, savgol_filter
from scipy.stats import ttest_1samp, pearsonr
import matplotlib.patches as patches
from visanalysis.util import plot_tools
from skimage import filters

from glom_pop import dataio, util


PROTOCOL_ID = 'ExpandingMovingSpot'
# PROTOCOL_ID = 'LoomingSpot'

if PROTOCOL_ID == 'ExpandingMovingSpot':
    eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
    key_value = ('diameter', 15.0)
elif PROTOCOL_ID == 'LoomingSpot':
    eg_series = ('2022-04-12', 2)  # ('2022-04-12', 2)
    key_value = ('rv_ratio', 100.0)


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
                                                  target_series_metadata={'protocol_ID': PROTOCOL_ID,
                                                                          'include_in_analysis': True,
                                                                          key_value[0]: key_value[1],
                                                                          },
                                                  target_groups=['aligned_response', 'behavior'])
# matching_series

# %%

fh0, ax0 = plt.subplots(1+len(included_gloms), 1, figsize=(6, 4))
[x.set_ylim([-0.15, 0.8]) for x in ax0.ravel()]
[util.clean_axes(x) for x in ax0.ravel()]
[x.set_ylim() for x in ax0.ravel()]
# [x.set_xlim([250, 500]) for x in ax0.ravel()]

fh1, ax1 = plt.subplots(1, 1, figsize=(3, 2))
ax1.set_axis_off()

fh2, ax2 = plt.subplots(1, 1, figsize=(6, 2))


corr_with_running = []
running_autocorr = []
gain_autocorr = []

response_amps = []
running_amps = []
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
    rmse_ds = resample(behavior_data['rmse'], response_data.get('response').shape[1])
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

    new_beh_corr = np.array([np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    # QC: check thresholding
    fh, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(rmse_ds)
    ax[0].axhline(thresh, color='r')
    ax[1].hist(rmse_ds, 100)
    ax[1].axvline(thresh, color='r')
    ax[0].set_title('{}: {}'.format(file_name, series_number))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        eg_trials = np.arange(30, 60)

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([running_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_behaving = np.concatenate([behavior_binary_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_running.shape[1]) * ID.getAcquisitionMetadata('sample_period')

        ax0[0].plot(concat_time, concat_running[0, :], color='k')
        ax0[0].set_ylim([concat_running.min(), concat_running.max()])
        ax0[0].set_ylabel('Movement', rotation=0)
        for g_ind, glom in enumerate(included_gloms):
            ax0[1+g_ind].set_ylabel(glom, rotation=0)
            ax0[1+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25)
            ax0[1+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[1+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[1+g_ind], dT=4, dF=0.25, T_value=0, F_value=-0.1)

        # Image of fly on ball:
        ax1.imshow(behavior_data['frame'], cmap='Greys_r')
        rect = patches.Rectangle((10, 90), behavior_data['frame'].shape[1]-30, behavior_data['frame'].shape[0]-90,
                                 linewidth=1, edgecolor='r', facecolor='none')

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

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RMS image difference')
fh0.savefig(os.path.join(save_directory, 'repeat_beh_resp_{}.svg'.format(PROTOCOL_ID)), transparent=True)
fh1.savefig(os.path.join(save_directory, 'repeat_beh_flyonball_{}.svg'.format(PROTOCOL_ID)), transparent=True)
fh2.savefig(os.path.join(save_directory, 'repeat_beh_running_{}.svg'.format(PROTOCOL_ID)), transparent=True)


# %% Summary plots: corr between trial gain and trial behavior

fh2, ax2 = plt.subplots(1, 1, figsize=(4, 2.5))
ax2.axhline(y=0, color='k', alpha=0.50)

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], 0, nan_policy='omit')
    p_vals.append(t_result.pvalue)

    if t_result.pvalue < (0.05 / len(included_gloms)):
        ax2.annotate('*', (g_ind, 0.45), fontsize=12)

    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax2.plot(g_ind * np.ones(corr_with_running.shape[0]), corr_with_running[:, g_ind],
             marker='.', color=util.get_color_dict()[glom], linestyle='none', alpha=0.5)

    ax2.plot(g_ind, y_mean,
             marker='o', color=util.get_color_dict()[glom])

    ax2.plot([g_ind, g_ind], [y_mean-y_err, y_mean+y_err],
             color=util.get_color_dict()[glom])

ax2.set_ylim([-0.9, 0.5])


ax2.set_xticks(np.arange(0, len(included_gloms)))
ax2.set_xticklabels(included_gloms, rotation=90)
ax2.set_ylabel('Corr. with behavior (r)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


# Plot behavior corr vs: (1) fraction time spent moving and (2) glom response amplitude
fh3, ax3 = plt.subplots(1, 2, figsize=(4.5, 2.0))
[x.set_ylim([-0.55, 0.05]) for x in ax3]

mean_glom_response_amplitude = np.nanmean(response_amps, axis=(1, 2))
mean_glom_corr_with_behavior = np.nanmean(corr_with_running, axis=0)
r = np.corrcoef(mean_glom_response_amplitude, mean_glom_corr_with_behavior)[0, 1]
print(r)
ax3[0].scatter(mean_glom_response_amplitude,
               mean_glom_corr_with_behavior,
               marker='o',
               c=list(util.get_color_dict().values()))

ax3[0].set_ylabel('Corr. with behavior')
ax3[0].set_xlabel('Mean response amplitude')
#
# ax3[1].plot(fraction_behaving, np.nanmean(corr_with_running, axis=1), 'ko')
# ax3[1].set_ylabel('Corr. with behavior')
# ax3[1].set_xlabel('Fraction of time behaving')
# ax3[1].spines['top'].set_visible(False)
# ax3[1].spines['right'].set_visible(False)
# r = np.corrcoef(fraction_behaving, np.nanmean(corr_with_running, axis=1))[0, 1]
# print('r = {}'.format(r))


fh2.savefig(os.path.join(save_directory, 'repeat_beh_summary_{}.svg'.format(PROTOCOL_ID)), transparent=True)
fh3.savefig(os.path.join(save_directory, 'repeat_beh_totalrunning_{}.svg'.format(PROTOCOL_ID)), transparent=True)


# %%

epoch_response_matrix.shape
running_response_matrix.shape
behavior_binary_matrix.shape

pre_frames = int(ID.getRunParameters('pre_time') / ID.getAcquisitionMetadata('sample_period'))
stim_frames = int(ID.getRunParameters('stim_time') / ID.getAcquisitionMetadata('sample_period'))

running_amp
fh, ax = plt.subplots(len(included_gloms), 1, figsize=(1, 12))
for g_ind, glom in enumerate(included_gloms):
    running_amp = running_response_matrix[0, ...].mean(axis=1)

    pre_running = running_response_matrix[0, :, 0:pre_frames].mean(axis=1)
    stim_running = running_response_matrix[0, :, pre_frames:(pre_frames+stim_frames)].mean(axis=1)

    response_amp = ID.getResponseAmplitude(epoch_response_matrix[g_ind, ...], metric='max')
    r, p = pearsonr(stim_running, response_amp)
    ax[g_ind].plot(stim_running, response_amp, 'k.')
    ax[g_ind].set_title(r)

epoch_response_matrix.shape
epoch_response_matrix.mean(axis=1).shape

peak_inds = np.argmax(epoch_response_matrix.mean(axis=1), axis=1)
peak_inds.shape
peak_inds
# %%
# Temporal relationship:
pre_running
stim_running
plt.plot(stim_running, pre_running, 'k.')

# %%


# %%

a = np.zeros(100)
a[42:45] = 1

b = np.zeros(100)
b[47:50] = 1

xc = util.getXcorr(a, b)

fh, ax = plt.subplots(1, 1, figsize=(6, 3))
xx = np.arange(len(a)) - len(a) / 2
ax.plot(xx, xc, 'k-o')
ax.set_xlim([-10, 10])



# %%
