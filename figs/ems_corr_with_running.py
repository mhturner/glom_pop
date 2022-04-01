import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data
from scipy.signal import resample
from scipy.stats import ttest_1samp
import matplotlib.patches as patches
from visanalysis.util import plot_tools

from glom_pop import dataio, util

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
data_directory = os.path.join(sync_dir, 'datafiles')
video_dir = os.path.join(sync_dir, 'behavior_videos')
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

eg_ind = 0
datasets = [('20220318', 8),
            ('20220324', 1),
            ('20220324', 7),
            ('20220324', 12),
            ('20220324', 16),
            ]


def getXcorr(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'same')
    return c


fh0, ax0 = plt.subplots(1+len(included_gloms), 1, figsize=(6, 4))
[x.set_ylim([-0.15, 0.8]) for x in ax0.ravel()]
[util.clean_axes(x) for x in ax0.ravel()]
[x.set_ylim() for x in ax0.ravel()]
# [x.set_xlim([250, 500]) for x in ax0.ravel()]

fh1, ax1 = plt.subplots(1, 1, figsize=(3, 2))
ax1.set_axis_off()

corr_with_running = []
running_autocorr = []
gain_autocorr = []
xcorr = []
for d_ind, ds in enumerate(datasets):
    series_number = ds[1]
    file_name = '{}-{}-{}.hdf5'.format(ds[0][:4], ds[0][4:6], ds[0][6:])

    # For video:
    series_dir = 'series' + str(series_number).zfill(3)
    date_dir = ds[0]
    file_path = os.path.join(data_directory, file_name)
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get video data:
    # Timing command from voltage trace
    voltage_trace, _, voltage_sample_rate = ID.getVoltageData()
    frame_triggers = voltage_trace[0, :]  # First voltage trace is trigger out

    video_filepath = glob.glob(os.path.join(video_dir, date_dir, series_dir) + "/*.avi")[0]  # should be just one .avi in there
    video_results = dataio.get_ball_movement(video_filepath,
                                             frame_triggers,
                                             sample_rate=voltage_sample_rate)

    # Show cropped ball and overall movement trace for QC
    fh_tmp, ax_tmp = plt.subplots(1, 2, figsize=(12, 3))
    ax_tmp[0].plot(video_results['frame_times'], video_results['rmse'], 'k')
    ax_tmp[1].imshow(video_results['cropped_frame'], cmap='Greys_r')
    ax_tmp[0].set_title(ds)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    # Resample to imaging rate and plot

    err_rmse_ds = resample(video_results['rmse'], response_data.get('response').shape[1])  # DO this properly based on response

    # Align running responses
    _, running_response_matrix = ID.getEpochResponseMatrix(err_rmse_ds[np.newaxis, :], dff=False)

    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    running_amp = ID.getResponseAmplitude(running_response_matrix, metric='mean')

    # Trial cross correlation between running and response amp (gain)
    running_autocorr.append(getXcorr(running_amp[0, :], running_amp[0, :]))
    gain_autocorr.append(np.vstack([getXcorr(response_amp[g_ind, :], response_amp[g_ind, :]) for g_ind in np.arange(len(included_gloms))]))
    xcorr.append(np.vstack([getXcorr(response_amp[g_ind, :], running_amp[0, :]) for g_ind in np.arange(len(included_gloms))]))

    new_beh_corr = np.array([np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    if d_ind == eg_ind:
        eg_trials = np.arange(0, 30)

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([running_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_running.shape[1]) * ID.getAcquisitionMetadata('sample_period')

        ax0[0].plot(concat_time, concat_running[0, :], color='k')
        ax0[0].set_ylim([err_rmse_ds.min(), err_rmse_ds.max()])
        ax0[0].set_ylabel('Movement', rotation=0)
        for g_ind, glom in enumerate(included_gloms):
            ax0[1+g_ind].set_ylabel(glom, rotation=0)
            ax0[1+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            ax0[1+g_ind].set_ylim([-0.1, 0.8])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[1+g_ind], dT=4, dF=0.25, T_value=0, F_value=-0.1)

        # Image of fly on ball:
        ax1.imshow(video_results['frame'], cmap='Greys_r')
        rect = patches.Rectangle((10, 90), video_results['frame'].shape[1]-30, video_results['frame'].shape[0]-90,
                                 linewidth=1, edgecolor='r', facecolor='none')
        # ax1.add_patch(rect)

fh0.savefig(os.path.join(save_directory, 'ems_running_eg.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'ems_running_flyonball.svg'), transparent=True)

corr_with_running = np.vstack(corr_with_running)  # flies x gloms

# %%
fh2, ax2 = plt.subplots(1, 1, figsize=(4, 2.5))
ax2.axhline(y=0, color='k', alpha=0.5)
p_vals = []

for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], 0, nan_policy='omit')
    p_vals.append(t_result.pvalue)

    if t_result.pvalue < (0.05 / len(included_gloms)):
        ax2.annotate('*', (g_ind, 0.45), fontsize=12)

    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax2.plot(g_ind * np.ones(corr_with_running.shape[0]), corr_with_running[:, g_ind],
             marker='.', color='k', linestyle='none', alpha=0.5)

    ax2.plot(g_ind, y_mean,
             marker='o', color=util.get_color_dict()[glom])

    ax2.plot([g_ind, g_ind], [y_mean-y_err, y_mean+y_err],
             color=util.get_color_dict()[glom])

    ax2.set_ylim([-0.6, 0.6])
ax2.set_xticks(np.arange(0, len(included_gloms)))
ax2.set_xticklabels(included_gloms, rotation=90)
ax2.set_ylabel('Corr. with behavior (r)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fh2.savefig(os.path.join(save_directory, 'ems_running_corr_summary.svg'), transparent=True)

# %% xcorr between behavior & responses

mean_xcorr = np.nanmean(np.stack(xcorr, axis=-1), axis=-1)
sem_xcorr = np.nanstd(np.stack(xcorr, axis=-1), axis=-1) / np.sqrt(len(xcorr))
xx = np.arange(0, mean_xcorr.shape[1]) - mean_xcorr.shape[1]/2

radius = 8
fh3, ax3 = plt.subplots(len(included_gloms), 1, figsize=(2, 4))
[x.set_xlim(-radius-0.2, radius+0.2) for x in ax3]
[x.set_ylim(-0.5, 0.15) for x in ax3]
[x.spines['top'].set_visible(False) for x in ax3]
[x.spines['right'].set_visible(False) for x in ax3]
[x.set_xticks([]) for x in ax3[:-1]]
[x.set_yticks([]) for x in ax3[:-1]]
ax3[-1].set_xlabel('Lag (trials)')
ax3[-1].set_ylabel('Correlation')
for g_ind, glom in enumerate(included_gloms):
    ax3[g_ind].axhline(y=0, color='k', alpha=0.5)
    ax3[g_ind].errorbar(xx, mean_xcorr[g_ind, :], yerr=sem_xcorr[g_ind, :], fmt='.-', color=util.get_color_dict()[glom])

fh3.savefig(os.path.join(save_directory, 'ems_running_xcorr.svg'), transparent=True)
# %%






# %% response - triggered walking?

# %% summary








# %%
