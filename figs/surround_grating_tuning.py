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
included_gloms
# Included only small spot responder gloms
included_vals = dataio.get_glom_vals_from_names(included_gloms)

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

# %%
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
eg_glom = 12

print(included_gloms[eg_glom])

# fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(1.5, 2.25))
fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(5, 6))
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
# fh0.savefig(os.path.join(save_directory, 'surround_grating_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)

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


# %%
fh2, ax2 = plt.subplots(len(included_gloms), 2, figsize=(3.5, 6.5))
[x.set_ylim([0, 1.1]) for x in ax2.ravel()]

for g_ind, glom in enumerate(included_gloms):
    glom_data = response_amps[g_ind, :, :]
    # norm within each fly
    glom_data_norm = glom_data / np.nanmax(glom_data, axis=(0, 1))[np.newaxis, np.newaxis, :]

    period_mean = np.nanmean(glom_data_norm, axis=(0, 2))
    period_err = np.nanstd(glom_data_norm, axis=(0, 2)) / np.sqrt(glom_data.shape[-1])

    rate_mean = np.nanmean(glom_data_norm, axis=(1, 2))
    rate_err = np.nanstd(glom_data_norm, axis=(1, 2)) / np.sqrt(glom_data.shape[-1])

    ax2[g_ind, 0].errorbar(x=target_grate_rates, y=rate_mean,
                         yerr=rate_err,
                         marker='None', linestyle='-', linewidth=2, color=util.get_color_dict()[glom], alpha=0.75)

    ax2[g_ind, 1].errorbar(x=target_grate_periods, y=period_mean,
                         yerr=period_err,
                         marker='None', linestyle='-', linewidth=2, color=util.get_color_dict()[glom], alpha=0.75)


    if g_ind == 12:
        ax2[g_ind, 0].set_xticks([20, 40, 80, 160, 320])
        ax2[g_ind, 0].set_xticklabels([20, '', 80, 160, 320])
        ax2[g_ind, 0].set_yticks([0, 1.0])
        ax2[g_ind, 0].set_xlabel('Speed ($\degree$/sec)')


        ax2[g_ind, 1].set_xticks([5, 10, 20, 40])
        ax2[g_ind, 1].set_xticklabels([5, 10, 20, 40])
        ax2[g_ind, 1].set_yticks([])
        ax2[g_ind, 1].set_xlabel('Period ($\degree$)')
    else:
        ax2[g_ind, 0].set_xticks([])
        ax2[g_ind, 0].set_yticks([])

        ax2[g_ind, 1].set_xticks([])
        ax2[g_ind, 1].set_yticks([])


    ax2[g_ind, 0].spines['top'].set_visible(False)
    ax2[g_ind, 0].spines['right'].set_visible(False)
    ax2[g_ind, 1].spines['top'].set_visible(False)
    ax2[g_ind, 1].spines['right'].set_visible(False)

fh2.supylabel('Response amplitude (normalized)')
fh2.savefig(os.path.join(save_directory, 'surround_grating_tuning.svg'), transparent=True)


# %%



# %%
