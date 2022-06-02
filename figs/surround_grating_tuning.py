from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd
import ants


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')


leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)


# %% (1) Vary rate and period


matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SurroundGratingTuning',
                                                                          'include_in_analysis': True,
                                                                          'grate_period': [5, 10, 20, 40],
                                                                          'grate_rate': [20, 40, 80, 160, 320],
                                                                          # 'spot_speed': [-100, 100]
                                                                          })

target_grate_rates = [20.,  40.,  80., 160., 320.]
target_grate_periods = [5., 10., 20., 40.]

all_responses = []
response_amps = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    print('-----')
    print('File = {} / series = {}'.format(file_name, series_number))

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    trial_averages = np.zeros((len(included_gloms), len(target_grate_rates), len(target_grate_periods), epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for gr_ind, gr in enumerate(target_grate_rates):
        for gp_ind, gp in enumerate(target_grate_periods):
            erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                       ID,
                                                                       query={'current_grate_rate': gr,
                                                                              'current_grate_period': gp},
                                                                       return_inds=True)

            trial_averages[:, gr_ind, gp_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x params x time
    all_responses.append(trial_averages)
    response_amps.append(ID.getResponseAmplitude(trial_averages))
    print('-----')
    if False:  # plot individual fly responses, QC
        fh, ax = plt.subplots(len(target_grate_rates), len(target_grate_periods), figsize=(8, 4))
        fh.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax.ravel()]
        [x.set_ylim([-0.1, 0.75]) for x in ax.ravel()]
        for gr_ind, gr in enumerate(target_grate_rates):
            for gp_ind, gp in enumerate(target_grate_periods):
                for g_ind, glom in enumerate(included_gloms):
                    ax[gr_ind, gp_ind].plot(trial_averages[g_ind, gr_ind, gp_ind, :], color=util.get_color_dict()[glom])
                if gr_ind == 0:
                    ax[gr_ind, gp_ind].set_title('{:.0f}'.format(gp))
                if gp_ind == 0:
                    ax[gr_ind, gp_ind].set_ylabel('{:.0f}'.format(gr))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

# %%  Mean across animals for eg glomerulus
eg_glom = 0
print(included_gloms[eg_glom])

fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(1.5, 2.25))
# fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(5, 6))
[x.spines['bottom'].set_visible(False) for x in ax0.ravel()]
[x.spines['left'].set_visible(False) for x in ax0.ravel()]
[x.spines['right'].set_visible(False) for x in ax0.ravel()]
[x.spines['top'].set_visible(False) for x in ax0.ravel()]
[x.set_xticks([]) for x in ax0.ravel()]
[x.set_yticks([]) for x in ax0.ravel()]
[x.set_ylim([-0.15, 0.75]) for x in ax0.ravel()]

for gr_ind, gr in enumerate(target_grate_rates):
    for gp_ind, gp in enumerate(target_grate_periods):
        tf = gr / gp  # deg/sec / deg = 1/sec
        lbl = '{:.0f} Hz'.format(tf) if (gr_ind+gp_ind) == 0 else '{:.1f}'.format(tf).rstrip('0').rstrip('.')
        ax0[gp_ind, gr_ind].annotate(lbl, (0.5, 0.4), ha='center', color=[0.5, 0.5, 0.5], fontsize=8)

        ax0[gp_ind, gr_ind].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
        ax0[gp_ind, gr_ind].plot(response_data['time_vector'],
                                mean_responses[eg_glom, gr_ind, gp_ind, :],
                                color=util.get_color_dict()[included_gloms[eg_glom]])

        ax0[gp_ind, gr_ind].fill_between(response_data['time_vector'],
                                        mean_responses[eg_glom, gr_ind, gp_ind, :] - sem_responses[eg_glom, gr_ind, gp_ind, :],
                                        mean_responses[eg_glom, gr_ind, gp_ind, :] + sem_responses[eg_glom, gr_ind, gp_ind, :],
                                        color=util.get_color_dict()[included_gloms[eg_glom]],
                                        alpha=0.5)


        if gp_ind == 0:
            ax0[gp_ind, gr_ind].set_title('{:.0f}'.format(gr), fontsize=8)
        if gr_ind == 0:
            ax0[gp_ind, gr_ind].set_ylabel('${:.0f}\degree$'.format(gp), fontsize=8)

        if np.logical_and(gr_ind == 0, gp_ind == 0):
            plot_tools.addScaleBars(ax0[gp_ind, gr_ind], dT=2, dF=0.25, T_value=-1, F_value=-0.1)

fh0.supylabel('Spatial period ($\degree$)')
fh0.suptitle('Speed ($\degree$/s)')
fh0.savefig(os.path.join(save_directory, 'surround_grating_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)

# %% Schematic figs...
# glom map inset for highlighted glom
glom_mask_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')).numpy()
fh1, ax1 = plt.subplots(1, 1, figsize=(2, 1))
util.make_glom_map(ax=ax1,
                   glom_map=glom_mask_2_meanbrain,
                   z_val=None,
                   highlight_names=[included_gloms[eg_glom]])
ax1.set_axis_off()
fh1.savefig(os.path.join(save_directory, 'surround_grating_{}_glommap.svg'.format(included_gloms[eg_glom])), transparent=True)

# Sine grating image
sf = np.pi/2
xx = np.linspace(0, 2*np.pi, 200)
yy = ID.getRunParameters('grate_contrast') * np.sin(2 * np.pi * sf * xx)
img = np.repeat(yy[:, np.newaxis], 100, axis=-1)
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 0.75))
ax2.imshow(img.T, cmap='Greys_r', vmin=-1, vmax=+1)
ax2.set_axis_off()
fh2.savefig(os.path.join(save_directory, 'surround_grating_stim.svg'), transparent=True)

# TODO: Histogram of angular velocities from walking data

# %% heatmaps for all gloms
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

fh2, ax2 = plt.subplots(3, 3, figsize=(3, 4), tight_layout=True)
[x.set_axis_off() for x in ax2.ravel()]
cbar_ax = fh2.add_axes([.91, .3, .03, .4])
for g_ind, glom in enumerate(included_gloms):
    ax2[rows[g_ind], cols[g_ind]].set_axis_on()
    glom_data = response_amps[g_ind, :, :]
    df = pd.DataFrame(data=np.nanmean(glom_data, axis=-1), index=target_grate_rates, columns=target_grate_periods)
    if g_ind == 5:
        xticklabels = np.array(target_grate_rates).astype('int')
        yticklabels = np.array(target_grate_periods).astype('int')
    else:
        xticklabels = False
        yticklabels = False

    if g_ind == 0:
        cbar = True
        cbar_ax = cbar_ax
        cbar_kws={'label': 'Response (dF/F)'}

    else:
        cbar=False
    sns.heatmap(df.T, ax=ax2[rows[g_ind], cols[g_ind]],
                xticklabels=xticklabels, yticklabels=yticklabels,
                vmin=0, vmax=0.35, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws,
                rasterized=True, cmap='viridis')
    ax2[rows[g_ind], cols[g_ind]].set_title(glom, color=util.get_color_dict()[glom], fontsize=10, fontweight='bold')

ax2[2, 0].set_xlabel('Speed ($\degree$/sec)')
ax2[2, 0].set_ylabel('Period ($\degree$)')

fh2.savefig(os.path.join(save_directory, 'surround_grating_tuning.svg'), transparent=True)


# %% (2) VARY DIRECTION
target_angles = [0, 45, 90, 135, 180, 225, 270, 315]
eg_series = ('2022-05-24', 5)  # ('2022-05-24', 5)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SurroundGratingTuning',
                                                                          'include_in_analysis': True,
                                                                          'angle': target_angles
                                                                          })



all_responses = []
response_amps = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)
    behaving_trials = np.where(behavior_data.get('behaving'))[0]
    nonbehaving_trials = np.where(~behavior_data.get('behaving'))[0]

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    trial_averages = np.zeros((len(included_gloms), len(target_angles), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for ang_ind, ang in enumerate(target_angles):
        erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                   ID,
                                                                   query={'current_angle': ang},
                                                                   return_inds=True)

        behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, ang_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, ang_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time

        # trial_averages[:, ang_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x params x time
    all_responses.append(trial_averages)
    response_amps.append(ID.getResponseAmplitude(trial_averages))

    # if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    if True: # QC
        fh0, ax0 = plt.subplots(len(included_gloms), len(target_angles), figsize=(4, 6))
        fh0.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax0.ravel()]
        [x.set_ylim([-0.1, 0.4]) for x in ax0.ravel()]
        for ang_ind, ang in enumerate(target_angles):
            for g_ind, glom in enumerate(included_gloms):
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 0,  :], color='b')
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 1, :], color='k')
            if g_ind == 0:
                ax0[gr_ind, gp_ind].set_title('{:.0f}'.format(ang))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)
# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

# %%  Polar plots, mean across animals
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

fh1, ax1 = plt.subplots(3, 3, figsize=(4, 5), subplot_kw={'projection': 'polar'})
[x.set_axis_off() for x in ax1.ravel()]

mean_tuning = []
for f_ind in range(len(matching_series)):
    fly_tuning = []
    for b in range(2):
        dir_tuning = ID.getResponseAmplitude(all_responses[:, :, b, :, f_ind])
        fly_tuning.append(dir_tuning)

        for g_ind, glom in enumerate(included_gloms):
            ax1[rows[g_ind], cols[g_ind]].set_axis_on()
            dir_resp = dir_tuning[g_ind, :]
            plot_resp = np.append(dir_resp, dir_resp[0])
            plot_dir = np.append(target_angles, target_angles[0])
            if b == 0:
                color = 'b'
            elif b == 1:
                color = 'k'
            # ax1[g_ind].plot(np.deg2rad(plot_dir), plot_resp, color=color, alpha=0.5)

    mean_tuning.append(np.stack(fly_tuning, axis=-1))

mean_tuning = np.stack(mean_tuning, axis=-1)
mean_tuning = np.nanmean(mean_tuning, axis=-1) # glom x dir x beh/nonbeh

# Plot mean dir tuning across flies, for beh vs. nonbeh trials
for g_ind, glom in enumerate(included_gloms):
    plot_dir = np.append(target_angles, target_angles[0])
    # Behaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 0], mean_tuning[g_ind, 0, 0])
    ax1[rows[g_ind], cols[g_ind]].plot(np.deg2rad(plot_dir), plot_resp,
                    color='b', linewidth=2, marker='.',
                    label='Behaving' if (g_ind == 0) else None)

    # Nonbehaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 1], mean_tuning[g_ind, 0, 1])
    ax1[rows[g_ind], cols[g_ind]].plot(np.deg2rad(plot_dir), plot_resp,
                    color='k', linewidth=2, marker='.',
                    label='Nonbehaving' if (g_ind == 0) else None)


    ax1[rows[g_ind], cols[g_ind]].annotate(glom, (0, 0), ha='center')
fh1.legend()
# %%
