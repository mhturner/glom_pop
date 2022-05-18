from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')


leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
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
# ID.getEpochParameters('current_grate_period')
target_grate_rates = [20.,  40.,  80., 160., 320.]
target_grate_periods = [5., 10., 20., 40.]

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

    if True:  # plot individual fly responses, QC
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

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

# %%  Mean across animals
eg_glom = 0
print(included_gloms[eg_glom])

fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(2, 2))
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


# glom map inset for highlighted glom
glom_mask_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')).numpy()
fh3, ax3 = plt.subplots(1, 1, figsize=(3, 2))
util.make_glom_map(ax=ax3,
                   glom_map=glom_mask_2_meanbrain,
                   z_val=None,
                   highlight_names=[included_gloms[eg_glom]])

# %%

mean_response_amp = ID.getResponseAmplitude(mean_responses)

fh1, ax1 = plt.subplots(len(included_gloms), 1, figsize=(0.4, 6))
cbar_ax = fh1.add_axes([.93, .3, .2, .25])
[x.set_axis_off() for x in ax1.ravel()]

fh2, ax2 = plt.subplots(len(included_gloms), 2, figsize=(2.5, 6.5))
[x.set_axis_off() for x in ax2.ravel()]
[x.set_ylim([0, 1.1]) for x in ax2.ravel()]

for g_ind, glom in enumerate(included_gloms):
    glom_data = mean_response_amp[g_ind, :, :]
    glom_data_norm = glom_data / glom_data.max()

    period_tuning = np.mean(glom_data_norm, axis=0)
    rate_tuning = np.mean(glom_data_norm, axis=1)

    ax2[g_ind, 0].axhline(0, color='k', alpha=0.5)
    ax2[g_ind, 0].plot(target_grate_rates, rate_tuning,
                       color=util.get_color_dict()[glom], marker='.')
    ax2[g_ind, 1].axhline(0, color='k', alpha=0.5)
    ax2[g_ind, 1].plot(target_grate_periods, period_tuning,
                       color=util.get_color_dict()[glom], marker='.')

    new_df = pd.DataFrame(data=glom_data_norm,
                          index=target_grate_rates, columns=target_grate_periods)
    if g_ind == 12:
        ax1[g_ind].set_axis_on()
        xticklabels = [5, 10, 20, 40]
        yticklabels = [20, '', 80, '', 320]


        ax2[g_ind, 0].set_axis_on()
        ax2[g_ind, 0].set_xticks([20, 40, 80, 160, 320])
        ax2[g_ind, 0].set_xticklabels([20, '', '', 160, 320])
        ax2[g_ind, 0].spines['top'].set_visible(False)
        ax2[g_ind, 0].spines['right'].set_visible(False)
        ax2[g_ind, 0].set_ylabel('Resp. (norm.)')
        ax2[g_ind, 0].set_xlabel('Speed\n($\degree$/sec)')

        ax2[g_ind, 1].set_axis_on()
        ax2[g_ind, 1].set_xticks([5, 10, 20, 40])
        ax2[g_ind, 1].set_xticklabels([5, '', '', 40])
        ax2[g_ind, 1].set_yticks([])
        ax2[g_ind, 1].spines['top'].set_visible(False)
        ax2[g_ind, 1].spines['right'].set_visible(False)
        ax2[g_ind, 1].set_xlabel('Period ($\degree$)')

    elif g_ind == 0:
        ax2[g_ind, 0].set_title('Speed')
        ax2[g_ind, 1].set_title('Spatial\nFrequency')

    else:
        xticklabels = False
        yticklabels = False

    sns.heatmap(new_df,
                cmap='viridis',
                vmin=0, vmax=1.0,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=g_ind == 0,
                cbar_kws={'label': 'Response peak (norm.)'},
                cbar_ax=None if g_ind else cbar_ax,
                ax=ax1[g_ind])
fh1.savefig(os.path.join(save_directory, 'surround_grating_heatmaps.svg'), transparent=True)
fh2.savefig(os.path.join(save_directory, 'surround_grating_tuning.svg'), transparent=True)

# %%




# %%
