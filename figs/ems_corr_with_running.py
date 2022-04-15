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

eg_ind = 8  # 8 (20220412, 5) - good, punctuated movement bouts
# Date, series, cropping for video
datasets = [
            ('20220318', 8, ((90, 0), (10, 20), (0, 0))),
            ('20220324', 1, ((100, 10), (10, 30), (0, 0))),
            ('20220324', 7, ((100, 10), (10, 30), (0, 0))),
            ('20220324', 12, ((100, 10), (10, 30), (0, 0))),
            ('20220324', 16, ((100, 10), (10, 30), (0, 0))),
            ('20220404', 1, ((120, 80), (150, 150), (0, 0))),
            ('20220404', 6, ((120, 80), (150, 150), (0, 0))),
            ('20220407', 2, ((120, 80), (150, 150), (0, 0))),
            ('20220412', 1, ((120, 60), (100, 100), (0, 0))),
            ('20220412', 5, ((120, 60), (100, 100), (0, 0))),
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

fh2, ax2 = plt.subplots(1, 1, figsize=(6, 2))

corr_with_running = []
running_autocorr = []
gain_autocorr = []
xcorr = []  # xxcorr is running, behavior. i.e. left of zero is behavior leads

erms = []
running_amps = []
fraction_behaving = []

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
                                             sample_rate=voltage_sample_rate,
                                             cropping=ds[2])

    # Show cropped ball and overall movement trace for QC
    fh_tmp, ax_tmp = plt.subplots(1, 2, figsize=(12, 3))
    tw_ax = ax_tmp[0].twinx()
    tw_ax.fill_between(video_results['frame_times'],
                       video_results['binary_behavior'],
                       color='k', alpha=0.5)
    ax_tmp[0].axhline(video_results['binary_thresh'], color='r')
    ax_tmp[0].plot(video_results['frame_times'],
                   video_results['rmse'],
                   'b')
    ax_tmp[1].imshow(video_results['cropped_frame'], cmap='Greys_r')
    ax_tmp[0].set_title(ds)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    erms.append(epoch_response_matrix)

    # Align running responses
    _, running_response_matrix = ID.getEpochResponseMatrix(resample(video_results['rmse'], response_data.get('response').shape[1])[np.newaxis, :],
                                                           dff=False)
    _, behavior_binary_matrix = ID.getEpochResponseMatrix(resample(video_results['binary_behavior'], response_data.get('response').shape[1])[np.newaxis, :],
                                                          dff=False)

    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    running_amp = ID.getResponseAmplitude(running_response_matrix, metric='mean')

    running_amps.append(running_amp)
    # Total fraction of all trials where animal behavior > binary threshold
    fraction_behaving.append(behavior_binary_matrix.sum() / behavior_binary_matrix.size)

    # Trial cross correlation between running and response amp (gain)
    running_autocorr.append(getXcorr(running_amp[0, :], running_amp[0, :]))
    gain_autocorr.append(np.vstack([getXcorr(response_amp[g_ind, :], response_amp[g_ind, :]) for g_ind in np.arange(len(included_gloms))]))
    xcorr.append(np.vstack([getXcorr(running_amp[0, :], response_amp[g_ind, :]) for g_ind in np.arange(len(included_gloms))]))

    new_beh_corr = np.array([np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    if d_ind == eg_ind:
        eg_trials = np.arange(0, 30)

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_running = np.concatenate([running_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_behaving = np.concatenate([behavior_binary_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_running.shape[1]) * ID.getAcquisitionMetadata('sample_period')

        ax0[0].plot(concat_time, concat_running[0, :], color='k')
        ax0[0].set_ylim([concat_running.min(), concat_running.max()])
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

        # Fly movement traj with thresh and binary shading
        tw_ax = ax2.twinx()
        tw_ax.fill_between(video_results['frame_times'],
                           video_results['binary_behavior'],
                           color=[0.5, 0.5, 0.5], alpha=0.5)
        ax2.axhline(video_results['binary_thresh'], color='r')
        ax2.plot(video_results['frame_times'],
                 video_results['rmse'],
                 'k')
        tw_ax.set_yticks([])

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('RMS image difference')
fh0.savefig(os.path.join(save_directory, 'ems_running_eg.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'ems_running_flyonball.svg'), transparent=True)
fh2.savefig(os.path.join(save_directory, 'ems_running_traj_eg.svg'), transparent=True)
corr_with_running = np.vstack(corr_with_running)  # flies x gloms

# %%
fh2, ax2 = plt.subplots(1, 1, figsize=(4, 2.5))
ax2.axhline(y=0, color='k', alpha=0.50)

fh3, ax3 = plt.subplots(1, 1, figsize=(2.5, 2.0))

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

ax2.set_ylim([-0.9, 0.5])


ax2.set_xticks(np.arange(0, len(included_gloms)))
ax2.set_xticklabels(included_gloms, rotation=90)
ax2.set_ylabel('Corr. with behavior (r)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


# Plot fraction time spent moving vs. correlation with behavior
ax3.plot(fraction_behaving, np.nanmean(corr_with_running, axis=1), 'ko')
ax3.set_ylabel('Corr. with behavior')
ax3.set_xlabel('Fraction of time behaving')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
r = np.corrcoef(fraction_behaving, np.nanmean(corr_with_running, axis=1))[0, 1]
print('r = {}'.format(r))

fh2.savefig(os.path.join(save_directory, 'ems_running_corr_summary.svg'), transparent=True)
# %%

# %%



# %% xcorr between behavior & responses

mean_xcorr = np.nanmean(np.stack(xcorr, axis=-1), axis=-1)
sem_xcorr = np.nanstd(np.stack(xcorr, axis=-1), axis=-1) / np.sqrt(len(xcorr))
xx = np.arange(0, mean_xcorr.shape[1]) - mean_xcorr.shape[1]/2

radius = 3
fh3, ax3 = plt.subplots(1, 13, figsize=(11, 1.5))
ax3 = ax3.ravel()
[x.set_xlim(-radius-0.2, radius+0.2) for x in ax3]
[x.set_ylim(-0.45, 0.15) for x in ax3]
[x.spines['top'].set_visible(False) for x in ax3]
[x.spines['right'].set_visible(False) for x in ax3]
[x.set_xticks([]) for x in ax3[1:]]
[x.set_yticks([]) for x in ax3[1:]]
ax3[0].set_xlabel('Lag (trials)')
ax3[0].set_ylabel('Correlation')
for g_ind, glom in enumerate(included_gloms):
    ax3[g_ind].axhline(y=0, color='k', alpha=0.5)
    ax3[g_ind].errorbar(xx, mean_xcorr[g_ind, :], yerr=sem_xcorr[g_ind, :], fmt='.-', color=util.get_color_dict()[glom])

fh3.savefig(os.path.join(save_directory, 'ems_running_xcorr.svg'), transparent=True)

# %% PCA on covariance matrix, and correlate PC projections with behavior


def getPCs(data_matrix):
    """
    data_matrix shape = gloms x features (e.g. gloms x time, or gloms x response amplitudes)
    """

    mean_sub = data_matrix - data_matrix.mean(axis=1)[:, np.newaxis]
    C = np.cov(mean_sub)
    evals, evecs = np.linalg.eig(C)

    # For modes where loadings are all negative, swap the sign
    for m in range(evecs.shape[1]):
        if np.all(np.sign(evecs[:, m]) <= 0):
            evecs[:, m] = -evecs[:, m]

    frac_var = evals / evals.sum()

    results_dict = {'eigenvalues': evals,
                    'eigenvectors': evecs,
                    'frac_var': frac_var}

    return results_dict

# %%


# Cut 4: LC26, 10: LC17
included_glom_inds = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12])
pca_gloms = included_gloms[included_glom_inds]
print('Including {}'.format(pca_gloms))

concat_all = np.concatenate(erms, axis=1)[included_glom_inds, :, :]
all_resp = ID.getResponseAmplitude(concat_all, metric='max')
all_resp[np.isnan(all_resp)] = 0

all_running = np.concatenate(running_amps, axis=1)

rd = getPCs(all_resp)

fh, ax = plt.subplots(1, 2, figsize=(4, 1.75))
ax[0].plot(rd['frac_var'], 'k-o')
ax[0].set_ylabel('Frac. var')
ax[0].set_xlabel('Component')

ax[1].bar(np.arange(len(included_glom_inds)), rd['eigenvectors'][:, 0])
ax[1].set_title('PC {}'.format(0))
ax[1].set_xticks(np.arange(len(included_glom_inds)))
ax[1].set_xticklabels(pca_gloms, rotation=90, fontsize=9)


F = rd['eigenvectors'] @ all_resp
fh, ax = plt.subplots(2, 1, figsize=(4, 3))
ax[0].plot(all_running[0, :], 'k')
ax[0].set_ylabel('Movement')
ax[1].plot(F[0, :])
ax[1].set_ylabel('PC 1 projection')
ax[1].set_xlabel('Trial')

fh, ax = plt.subplots(1, 1, figsize=(2, 2))
xx = np.arange(0, all_running.shape[1]) - all_running.shape[1]/2  # For xcorr
cc = getXcorr(all_running[0, :], F[0, :])
ax.plot(xx, cc, 'k-o')
ax.set_xlim([-15, 15])
ax.set_ylim([-0.4, 0.25])
ax.axhline(0, color='k', alpha=0.5)



# %%


# %%

fh, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.plot(all_running[0, :], 'k')
tx = ax.twinx()
tx.plot(F[0, :], 'b')



# %% IND FLIES

# Cut 4: LC26, 10: LC17
included_glom_inds = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12])
pca_gloms = included_gloms[included_glom_inds]
print('Including {}'.format(pca_gloms))

fh, ax = plt.subplots(len(erms), 3, figsize=(12, 4))
r_vals = []
xcorrs = []
for f_ind, erm in enumerate(erms):
    fly_response = ID.getResponseAmplitude(erm, metric='max')[included_glom_inds, :]
    fly_response[np.isnan(fly_response)] = 0
    fly_running = running_amps[f_ind]

    pca_results = getPCs(fly_response)
    ax[f_ind, 0].plot(pca_results['frac_var'], 'k-o')

    # First mode
    ax[f_ind, 1].bar(np.arange(len(included_glom_inds)), pca_results['eigenvectors'][:, 0])

    # Projection of PC onto data:
    F = pca_results['eigenvectors'] @ fly_response

    ax[f_ind, 2].plot(fly_running[0, :], color='k', alpha=0.5)
    ax2 = ax[f_ind, 2].twinx()
    ax2.plot(F[0, :], color='b', alpha=1.0)
    ax2.set_yticks([])
    cc = getXcorr(fly_running[0, :], F[0, :])
    xcorrs.append(cc)

    r = np.corrcoef(fly_running[0, :], F[0, :], )[0, 1]
    r_vals.append(r)


xcorrs = np.vstack(xcorrs)
# %%

r_vals
xx = np.arange(0, xcorrs.shape[1]) - xcorrs.shape[1]/2
mean_xcorr = np.nanmean(xcorrs, axis=0)
sem_xcorr = np.nanstd(xcorrs, axis=0) / np.sqrt(xcorrs.shape[0])

fh, ax = plt.subplots(1, 1, figsize=(2, 3))
ax.plot(xx, mean_xcorr, 'k-o')
ax.set_xlim([-20, 20])
ax.axhline(0, color='k', alpha=0.5)
ax.fill_between(xx, mean_xcorr-sem_xcorr, mean_xcorr+sem_xcorr, color='k', alpha=0.5)


# %% (1) TCA


import tensortools as tt

# Single fly eg
# eg_ind = 1
# eg_response = np.swapaxes(erms[eg_ind], 1, 2)
# eg_running = running_amps[eg_ind][0, :]
# eg_response[np.isnan(eg_response)] = 0


# Concat all trials across animals
concat_all = np.concatenate(erms, axis=1)[included_glom_inds, :, :]
running_data = np.squeeze(np.concatenate(running_amps, axis=-1))
data = np.swapaxes(concat_all, 1, 2)

# Data shape = (gloms x time x trials)
print('Data shape = {}'.format(data.shape))
# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
ensemble = tt.Ensemble(fit_method="ncp_bcd")
ensemble.fit(data, ranks=range(1, 9), replicates=4)

# %%
fig, axes = plt.subplots(1, 2)
tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
fig.tight_layout()

num_components = 5
replicate = 2
U = ensemble.factors(num_components)[replicate]

glom_factors = U[0]
temporal_factors = U[1]
trial_factors = U[2]

print('glom_factors: {}\n temporal_factors:{}\n trial_factors:{}'.format(glom_factors.shape,
                                                                         temporal_factors.shape,
                                                                         trial_factors.shape,
                                                                         ))
# %%
colors = 'rgbcm'
fh, ax = plt.subplots(num_components+1, 3, figsize=(6, 3))
ax[0, 0].plot(running_data, 'k')
[ax[x+1, 0].plot(trial_factors[:, x],
                 color=colors[x]) for x in range(num_components)]

[ax[x+1, 1].plot(temporal_factors[:, x],
                 color=colors[x]) for x in range(num_components)]

[ax[x+1, 2].bar(np.arange(len(included_glom_inds)), glom_factors[:, x],
                color=colors[x]) for x in range(num_components)]
# %%

r_vals = [np.corrcoef(running_data, trial_factors[:, x])[0, 1] for x in range(num_components)]
fh, ax = plt.subplots(1, num_components, figsize=(6, 2))
[ax[x].set_title('r = {:.2f}'.format(r_vals[x])) for x in range(num_components)]
[ax[x].plot(running_data, trial_factors[:, x],
            color=colors[x],
            linestyle='none',
            marker='.') for x in range(num_components)]






# %%





# %%
