from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel


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

# %%

target_grate_rates
fh, ax = plt.subplots(len(target_grate_rates), len(target_grate_periods), figsize=(5, 5))
# [plot_tools.cleanAxes(x) for x in ax.ravel()]
[x.spines['bottom'].set_visible(False) for x in ax.ravel()]
[x.spines['left'].set_visible(False) for x in ax.ravel()]
[x.spines['right'].set_visible(False) for x in ax.ravel()]
[x.spines['top'].set_visible(False) for x in ax.ravel()]
[x.set_xticks([]) for x in ax.ravel()]
[x.set_yticks([]) for x in ax.ravel()]
[x.set_ylim([-0.1, 0.75]) for x in ax.ravel()]
fh.suptitle('Period (Deg.)')
fh.supylabel('Speed (Deg./sec.)')
for gr_ind, gr in enumerate(target_grate_rates):
    for gp_ind, gp in enumerate(target_grate_periods):
        tf = gr / gp  # deg/sec / deg = 1/sec
        lbl = '{:.0f} Hz'.format(tf) if (gr_ind+gp_ind) == 0 else '{:.0f}'.format(tf)
        ax[gr_ind, gp_ind].annotate(lbl, (0.5, 0.4), ha='center', color=[0.5, 0.5, 0.5])
        for g_ind, glom in enumerate(included_gloms):
            ax[gr_ind, gp_ind].plot(response_data['time_vector'],
                                    mean_responses[g_ind, gr_ind, gp_ind, :],
                                    color=util.get_color_dict()[glom])
        if gr_ind == 0:
            ax[gr_ind, gp_ind].set_title('{:.0f}'.format(gp))
        if gp_ind == 0:
            ax[gr_ind, gp_ind].set_ylabel('{:.0f}'.format(gr))



# %%
import seaborn as sns
import pandas as pd
response_amps.shape
grate_rates
# Normalize by response to slowest grating rate and lowest period (20 deg/sec and 5 deg)
norm = response_amps / response_amps[:, 0, 0, :][:, np.newaxis, np.newaxis, :]

np.nanmean(norm[g_ind, :, :, :], axis=-1).max()
fh, ax = plt.subplots(4, 4, figsize=(10, 10))
ax = ax.ravel()
for g_ind, glom in enumerate(included_gloms):
    new_df = pd.DataFrame(data=np.nanmean(norm[g_ind, :, :, :], axis=-1),
                          index=grate_rates, columns=grate_periods)
    sns.heatmap(new_df, cmap='RdBu_r', vmin=0, vmax=1.25, ax=ax[g_ind])


# %%
