from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel, pearsonr
from flystim import image
from scipy.interpolate import interp1d


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SaccadeSuppression',
                                                                          'include_in_analysis': True,
                                                                          'saccade_sample_period': 0.25,
                                                                          })

for series in np.array(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('{}: {}'.format(file_name, series_number))



# %% Stim schematic

fh2, ax2 = plt.subplots(1, 1, figsize=(2, 2))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
tt = np.linspace(0.01, 2.99, 100)
switch_times = [1, 1.5, 2]
for st_ind, st in enumerate(switch_times):
    t = [0, st, st+0.2, 3]
    p = [0, 0, 70, 70]
    interp_fh = interp1d(t, p, kind='linear')
    loc = interp_fh(tt)
    loc -= loc[0]  # start at 0
    ax2.plot(tt, loc, alpha=1, color=[0.5, 0.5, 0.5], linewidth=2)

ax2.quiver(1.6, 35, 1, 0, color='y', scale=3, width=0.02, headwidth=3)
ax2.quiver(1.6, 35, -1, 0, color='y', scale=3, width=0.02, headwidth=3)
ax2.annotate('$\Delta t$', (1.65, 40))
ax2.plot(1.6, 35, 'yo')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Background \nposition ($\degree$)')
fh2.savefig(os.path.join(save_directory, 'saccade_schematic.svg'), transparent=True)


# %% No behavior split:


target_saccade_times = np.arange(0, 3, 0.25)

all_response_gains = []
all_response_amps = []
all_responses = []
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

    trial_averages = np.zeros((len(included_gloms), len(target_saccade_times), epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for st_ind, st in enumerate(target_saccade_times):
        _, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                   ID,
                                                                   query={'current_saccade_time': st},
                                                                   return_inds=True)

        trial_averages[:, st_ind, :] = np.nanmean(epoch_response_matrix[:, matching_inds, :], axis=1)  # each trial average: gloms x time


    resp_amp = ID.getResponseAmplitude(trial_averages, metric='max')  # glom x saccade time
    # gain := response amp / amp for last saccade time. Norm is done *within each behavioral condition*
    norm_val = resp_amp[:, -1]  # last saccade time, basically at the end of the trial, well after the response has ended
    response_gain = resp_amp / norm_val[:, np.newaxis]  # gloms x saccade times
    all_response_gains.append(response_gain)
    all_responses.append(trial_averages)
    all_response_amps.append(resp_amp)

    if False: # Plot ind fly responses. QC
        eg_saccade_inds = np.arange(0, len(target_saccade_times), 1)
        fh, ax = plt.subplots(len(included_gloms), len(eg_saccade_inds), figsize=(8, 4))
        fh.suptitle('{}: {}'.format(file_name, series_number))
        [x.set_ylim([-0.1, 0.5]) for x in ax.ravel()]
        [x.set_axis_off() for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            for ind, si in enumerate(eg_saccade_inds):
                ax[g_ind, ind].plot(response_data['time_vector'], trial_averages[g_ind, si, :], color=util.get_color_dict()[glom])


all_response_gains = np.stack(all_response_gains, axis=-1)
all_response_amps = np.stack(all_response_amps, axis=-1)
all_responses = np.stack(all_responses, axis=-1)


# %% eg glom and fly traces
eg_fly_ind = 4
eg_glom_ind = 0    # 0, LC11; 3, LC6

eg_saccade_inds = np.arange(0, 12, 2)

fh1, ax1 = plt.subplots(1, len(eg_saccade_inds), figsize=(3, 1.5))
[x.set_ylim([-0.1, 0.5]) for x in ax1.ravel()]
[x.set_axis_off() for x in ax1.ravel()]
for ind, si in enumerate(eg_saccade_inds):
    xval = ID.getRunParameters('pre_time') + target_saccade_times[si]
    ax1[ind].plot([xval, xval], [0, 0.5], color='y', linestyle='-', alpha=1, linewidth=1, zorder=10)
    ax1[ind].axhline(y=0, color='k', alpha=0.5)
    ax1[ind].plot(response_data['time_vector'], all_responses[eg_glom_ind, si, :, eg_fly_ind],
                  color=util.get_color_dict()[included_gloms[eg_glom_ind]], alpha=1.0, linewidth=2)

    if ind == 0:
        plot_tools.addScaleBars(ax1[ind], dT=2, dF=0.25, T_value=-0.1, F_value=-0.08)

fh1.savefig(os.path.join(save_directory, 'saccade_traces_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)

# %%
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

# Response onset := peak slope of mean response across all stims, animals
onset_inds = [np.argmax(np.diff(np.mean(all_responses[x, :, :, :], axis=(0, 2)))) for x in range(len(included_gloms))]
tt = response_data['time_vector'] - ID.getRunParameters('pre_time')
# Norm by: last saccade response, nonbehaving condition
all_response_amps_normed = all_response_amps / all_response_amps[:, -1, :][:, np.newaxis, :]
popmean = np.nanmean(all_response_amps_normed, axis=-1)
poperr = np.nanstd(all_response_amps_normed, axis=-1) / np.sqrt(all_response_amps.shape[-1])
fh3, ax3 = plt.subplots(3, 3, figsize=(4, 3.5), tight_layout=True)
[x.set_axis_off() for x in ax3.ravel()]
[x.set_xlim([-2, 2]) for x in ax3.ravel()]
[x.set_ylim([0, 1.5]) for x in ax3.ravel()]
[x.set_xticks([-2, -1, 0, 1, 2]) for x in ax3.ravel()]
for g_ind, glom in enumerate(included_gloms):
    # indication of response onset time
    onset_time = tt[onset_inds[g_ind]]
    ax3[rows[g_ind], cols[g_ind]].axvline(x=0, color='k', alpha=0.5, linestyle='-', linewidth=1)
    ax3[rows[g_ind], cols[g_ind]].axhline(y=1, color='k', alpha=0.5)
    ax3[rows[g_ind], cols[g_ind]].errorbar(x=target_saccade_times-onset_time,
                                           y=popmean[g_ind, :],
                                           yerr=poperr[g_ind, :],
                                           color=util.get_color_dict()[glom],
                                           alpha=1.0)

    ax3[rows[g_ind], cols[g_ind]].set_axis_on()
    ax3[rows[g_ind], cols[g_ind]].set_title(glom)
    ax3[rows[g_ind], cols[g_ind]].set_ylim(bottom=0)
    ax3[rows[g_ind], cols[g_ind]].spines['top'].set_visible(False)
    ax3[rows[g_ind], cols[g_ind]].spines['right'].set_visible(False)


fh3.supylabel('Response gain')
fh3.supxlabel('Saccade time relative to response onset (s)')
fh3.savefig(os.path.join(save_directory, 'saccade_gain_summary.svg'), transparent=True)


 # %% BEHAVIOR SPLIT: for saccades near response onset

target_saccade_times = np.arange(0, 3, 0.25)

all_response_gains = []
all_response_amps = []
all_responses = []
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

    # Load behavior data
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             response_len=response_data.get('response').shape[1],
                                             process_behavior=True, fps=50, exclude_thresh=300)
    behaving_trials = np.where(behavior_data.get('is_behaving')[0])[0]
    nonbehaving_trials = np.where(~behavior_data.get('is_behaving')[0])[0]

    trial_averages = np.zeros((len(included_gloms), len(target_saccade_times), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for st_ind, st in enumerate(target_saccade_times):
        _, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                ID,
                                                                query={'current_saccade_time': st},
                                                                return_inds=True)
        behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, st_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, st_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time


        resp_amp = ID.getResponseAmplitude(trial_averages, metric='max')  # glom x saccade time x beh/nonbeh
        # gain := response amp / amp for last saccade time. Norm is done *within each behavioral condition*
        norm_val = resp_amp[:, -1, :]  # last saccade time, basically at the end of the trial, well after the response has ended
        response_gain = resp_amp / norm_val[:, np.newaxis, :]  # gloms x saccade times x beh/nonbeh
    all_response_gains.append(response_gain)
    all_responses.append(trial_averages)
    all_response_amps.append(resp_amp)

    if False: # Plot ind fly responses. QC
        eg_saccade_inds = np.arange(0, len(target_saccade_times), 1)
        fh, ax = plt.subplots(len(included_gloms), len(eg_saccade_inds), figsize=(8, 4))
        fh.suptitle('{}: {}'.format(file_name, series_number))
        [x.set_ylim([-0.1, 0.5]) for x in ax.ravel()]
        [x.set_axis_off() for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            for ind, si in enumerate(eg_saccade_inds):
                ax[g_ind, ind].plot(response_data['time_vector'], trial_averages[g_ind, si, 0, :], color=util.get_color_dict()[glom])
                ax[g_ind, ind].plot(response_data['time_vector'], trial_averages[g_ind, si, 1, :], color=util.get_color_dict()[glom], alpha=0.5)


all_response_gains = np.stack(all_response_gains, axis=-1)
all_response_amps = np.stack(all_response_amps, axis=-1)
all_responses = np.stack(all_responses, axis=-1)
# %%
eg_glom_ind = 0
eg_fly_ind = 4
# Eg glom: mean response to saccade suppression in walking vs. stationary
mean_resp = np.nanmean(all_responses[eg_glom_ind, 5:10, :, : , eg_fly_ind], axis=(0))  # beh/nonbeh x time
baseline_resp = all_responses[eg_glom_ind, -1, :, : , eg_fly_ind]
fh, ax = plt.subplots(1, 2, figsize=(2.75, 1.5))
[x.set_ylim([-0.1, 0.5]) for x in ax]
[util.clean_axes(x) for x in ax]
ax[0].axhline(y=0, color='k', alpha=0.5)
ax[0].plot(response_data['time_vector'], baseline_resp[0, :],
           alpha=1, color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2,
           label='Walking')
ax[0].plot(response_data['time_vector'], baseline_resp[1, :],
           alpha=0.5, color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2,
           label='Stationary')
ax[0].set_title('No visual\nsaccade')
plot_tools.addScaleBars(ax[0], dT=2, dF=0.25, T_value=-0.1, F_value=-0.08)

ax[1].axhline(y=0, color='k', alpha=0.5)
ax[1].plot(response_data['time_vector'], mean_resp[0, :],
           alpha=1, color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2)
ax[1].plot(response_data['time_vector'], mean_resp[1, :],
           alpha=0.5, color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2)
ax[1].set_title('With visual\nsaccade')

fh.legend()

fh.savefig(os.path.join(save_directory, 'saccade_beh_eg_traces.svg'), transparent=True)

# %% Summary beh vs nonbeh

# Norm by: last saccade response, nonbehaving condition
all_response_amps_normed = all_response_amps / all_response_amps[:, -1, 1, :][:, np.newaxis, np.newaxis, :]

fh3, ax3 = plt.subplots(3, 3, figsize=(4, 4), tight_layout=True)
[x.set_axis_off() for x in ax3.ravel()]
[x.spines['top'].set_visible(False) for x in ax3.ravel()]
[x.spines['right'].set_visible(False) for x in ax3.ravel()]
beh_suppression = []
visual_suppression = []



for g_ind, glom in enumerate(included_gloms):
    fly_beh = np.mean(all_response_amps_normed[g_ind, 5:10, 0, :], axis=(0))
    mean_beh = np.mean(fly_beh)
    err_beh = np.std(fly_beh) / np.sqrt(fly_beh.shape[-1])

    fly_nonbeh = np.mean(all_response_amps_normed[g_ind, 5:10, 1, :], axis=(0))
    mean_nonbeh = np.mean(fly_nonbeh)
    err_nonbeh = np.std(fly_nonbeh) / np.sqrt(fly_nonbeh.shape[-1])
    ymax = 1.1*np.max([fly_beh, fly_nonbeh])
    ymax = np.max([ymax, 1.1])

    h, p = ttest_rel(fly_beh, fly_nonbeh)
    if p < 0.05:
        ax3[rows[g_ind], cols[g_ind]].annotate(r'$\ast$', (0, 0.9*ymax), fontsize=12)

    ax3[rows[g_ind], cols[g_ind]].set_axis_on()
    ax3[rows[g_ind], cols[g_ind]].plot([0, ymax], [0, ymax], 'k--', alpha=0.5)
    ax3[rows[g_ind], cols[g_ind]].plot(fly_beh, fly_nonbeh, linestyle='None', marker='.', color=[0.5, 0.5, 0.5])
    ax3[rows[g_ind], cols[g_ind]].errorbar(x=mean_beh,
                                           y=mean_nonbeh,
                                           xerr=err_beh,
                                           yerr=err_nonbeh,
                                           linestyle='None', marker='o',
                                           color=util.get_color_dict()[glom])
    ax3[rows[g_ind], cols[g_ind]].set_title('{}'.format(glom))

    # suppression due to vis saccade and beh suppression
    new_visual_suppression = np.mean(1-fly_nonbeh)  # already normalized to non-saccade condition, by def'n of gain. So just sub from 1
    visual_suppression.append(new_visual_suppression)
    new_beh_suppression = np.mean((fly_nonbeh-fly_beh)/fly_nonbeh)  # relative effect of behavior suppression
    beh_suppression.append(new_beh_suppression)


fh3.supxlabel('Response gain, walking')
fh3.supylabel('Response gain, stationary')
fh3.savefig(os.path.join(save_directory, 'saccade_beh_summary.svg'), transparent=True)




# %%
# Corr between visual & behavioral suppression for each glom

# each is shape: gloms x flies
baseline = all_response_amps[:, -1, 1, :]  # Baseline, no vis no beh. Last saccade time, stationary
vis = all_response_amps[:, 6, 1, :]  # vis, no beh
beh = all_response_amps[:, -1, 0, :]  # beh, no vis
vis_beh = all_response_amps[:, 6, 0, :]  # beh + vis suppression


vis_gain = vis / baseline
beh_gain = beh / baseline
vis_beh_gain = vis_beh / baseline

fh4, ax4 = plt.subplots(1, 2, figsize=(5, 2.75), tight_layout=True)
ax4[0].spines['top'].set_visible(False)
ax4[0].spines['right'].set_visible(False)
ax4[0].plot([0.25, 1.3], [0.25, 1.3], color='k', alpha=0.5, linestyle='--')

r, p = pearsonr(np.mean(beh_gain, axis=-1), np.mean(vis_gain, axis=-1))
for g_ind, glom in enumerate(included_gloms):
    ax4[0].errorbar(x=np.mean(beh_gain, axis=-1)[g_ind],
                 y=np.mean(vis_gain, axis=-1)[g_ind],
                 xerr=np.std(beh_gain, axis=-1)[g_ind] / np.sqrt(beh_gain.shape[1]),
                 yerr=np.std(vis_gain, axis=-1)[g_ind] / np.sqrt(vis_gain.shape[1]),
                 marker='o', color=util.get_color_dict()[glom], label=glom)

print('r = {:.2f}'.format(r))
# [ax4[0].annotate(included_gloms[g_ind],
#               [np.mean(beh_gain, axis=-1)[g_ind]+0.02, np.mean(vis_gain, axis=-1)[g_ind]],
#               ha='left', va='center', fontsize=10)
#               for g_ind in range(len(included_gloms))]

ax4[0].set_ylim([0.4, 1.3])
ax4[0].set_xlim([0.4, 1.3])
ax4[0].set_xlabel('Gain while walking')
ax4[0].set_ylabel('Gain during visual saccade')
# fh4[0].savefig(os.path.join(save_directory, 'saccade_beh_vis_corr.svg'), transparent=True)
fh4.legend(ncol=3, fontsize=9)
ax4[0].set_title('Balance of \ngain mechanisms')



indep_prod = vis_norm * beh_norm
# fh, ax = plt.subplots(1, 1, figsize=(2.25, 2.25))
ax4[1].plot([0, 1.5], [0, 1.5], color='k', alpha=0.5, linestyle='--')
ax4[1].spines['top'].set_visible(False)
ax4[1].spines['right'].set_visible(False)
for g_ind, glom in enumerate(included_gloms):
    ax4[1].errorbar(x=np.mean(indep_prod[g_ind, :]),
                y=np.mean(vis_beh_norm[g_ind, :]),
                xerr=np.std(indep_prod[g_ind, :])/np.sqrt(indep_prod.shape[-1]),
                yerr=np.std(vis_beh_norm[g_ind, :])/np.sqrt(vis_beh_norm.shape[-1]),
                color=util.get_color_dict()[glom],
                marker='o',
                linewidth=2)
indep_prod = np.mean(vis_norm, axis=-1) * np.mean(beh_norm, axis=-1)

# [ax.annotate(included_gloms[g_ind],
#              [indep_prod[g_ind]+0.02, np.mean(vis_beh_norm, axis=-1)[g_ind]],
#              ha='left', va='center', fontsize=10)
#              for g_ind in range(len(included_gloms))];
ax4[1].set_xlabel('gain(Vis. only) x gain(Beh. only)')
ax4[1].set_ylabel('gain(Vis. & Beh.)')
ax4[1].set_title('Independence of \ngain mechanisms')

fh4.savefig(os.path.join(save_directory, 'saccade_beh_vis_indep.svg'), transparent=True)
# %%

# each is shape: gloms x flies
baseline = all_response_amps[:, -1, 1, :]  # Baseline, no vis no beh. Last saccade time, stationary
vis = all_response_amps[:, 6, 1, :]  # vis, no beh
beh = all_response_amps[:, -1, 0, :]  # beh, no vis
vis_beh = all_response_amps[:, 6, 0, :]  # beh + vis suppression

vis_norm = vis / baseline
beh_norm = beh / baseline
vis_beh_norm = vis_beh / baseline
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

fh5, ax5 = plt.subplots(3, 3, figsize=(4, 4), tight_layout=True)
[x.set_axis_off() for x in ax5.ravel()]
[x.set_xticks([]) for x in ax5.ravel()]
[x.set_yticks([]) for  x in ax5.ravel()]
[x.set_ylim([0, 1.5]) for x in ax5.ravel()]
[x.set_xlim([0.5, 3.5]) for x in ax5.ravel()]
[x.spines['top'].set_visible(False) for x in ax5.ravel()]
[x.spines['right'].set_visible(False) for x in ax5.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax5[rows[g_ind], cols[g_ind]].set_axis_on()
    ax5[rows[g_ind], cols[g_ind]].axhline(y=1, color=[0.5, 0.5, 0.5], alpha=0.5, linestyle='--')
    ax5[rows[g_ind], cols[g_ind]].errorbar(x=1,
                                           y=np.mean(vis_norm[g_ind, :]),
                                           yerr=np.std(vis_norm[g_ind, :])/np.sqrt(vis_norm.shape[-1]),
                                           marker='o', linewidth=2, color='r')
    ax5[rows[g_ind], cols[g_ind]].errorbar(x=2,
                                           y=np.mean(beh_norm[g_ind, :]),
                                           yerr=np.std(beh_norm[g_ind, :])/np.sqrt(beh_norm.shape[-1]),
                                           marker='o', linewidth=2, color='b')
    ax5[rows[g_ind], cols[g_ind]].errorbar(x=3,
                                           y=np.mean(vis_beh_norm[g_ind, :]),
                                           yerr=np.std(vis_beh_norm[g_ind, :])/np.sqrt(vis_beh_norm.shape[-1]),
                                           marker='o', linewidth=2, color='m')
    ax5[rows[g_ind], cols[g_ind]].set_title(glom)


[x.set_xticks([1, 2, 3]) for x in ax5[2, :]]
[x.set_xticklabels(['Vis.', 'Beh.', 'Vis.+Beh.'], rotation=90) for x in ax5[2, :]]

[x.set_yticks([0, 0.5, 1, 1.5]) for x in ax5.ravel()]

fh5.supylabel('Response gain')

fh5.savefig(os.path.join(save_directory, 'saccade_beh_vis_conditions.svg'), transparent=True)

# %% gain(vis+beh) vs. gain(vis) * gain(beh)



# %%
