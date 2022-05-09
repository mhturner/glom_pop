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

plot_glom_ind = 0
all_responses = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Align responses
    parameter_key = ['current_spot_speed', 'current_grate_rate', 'current_grate_period']
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix, parameter_key=parameter_key)

    all_responses.append(mean_response)


# %%

target_spot_speed = 100
pull_speed_ind = np.where([target_spot_speed == x[0] for x in unique_parameter_values])[0]

grate_rates = np.unique([x[1] for x in unique_parameter_values])
grate_periods = np.unique([x[2] for x in unique_parameter_values])

plot_glom_ind = 1
response_amps = np.zeros((len(included_gloms), len(grate_rates), len(grate_periods), len(all_responses)))
for fly_ind, mean_response in enumerate(all_responses):
    fly_resp_amps = ID.getResponseAmplitude(mean_response, metric='max')
    fh, ax = plt.subplots(len(grate_rates), len(grate_periods), figsize=(8, 4))
    [plot_tools.cleanAxes(x) for x in ax.ravel()]
    [x.set_ylim([-0.1, 0.75]) for x in ax.ravel()]
    for gr_ind, gr in enumerate(grate_rates):
        for gp_ind, gp in enumerate(grate_periods):
            pull_rate_ind = np.where([gr == x[1] for x in unique_parameter_values])[0]
            pull_period_ind = np.where([gp == x[2] for x in unique_parameter_values])[0]

            pull_ind = list(set.intersection(set(pull_rate_ind),
                                             set(pull_period_ind),
                                             set(pull_speed_ind)))

            assert len(pull_ind) == 1
            pull_ind = pull_ind[0]

            response_amps[:, gr_ind, gp_ind, fly_ind] = ID.getResponseAmplitude(mean_response[:, pull_ind, :])

            ax[gr_ind, gp_ind].plot(mean_response[plot_glom_ind, pull_ind, :], color='k')

            if gr_ind == 0:
                ax[gr_ind, gp_ind].set_title('{:.0f}'.format(gp))
            if gp_ind == 0:
                ax[gr_ind, gp_ind].set_ylabel('{:.0f}'.format(gr))

# %%
import seaborn as sns

# Normalize by response to slowest grating rate and lowest period (20 deg/sec and 5 deg)
norm = response_amps / response_amps[:, 0, 0, :][:, np.newaxis, np.newaxis, :]

fh, ax = plt.subplots(4, 4, figsize=(10, 10))
ax = ax.ravel()
for g_ind, glom in enumerate(included_gloms):
    sns.heatmap(np.nanmean(norm[g_ind, :, :, :], axis=-1), cmap='Reds', vmin=0, vmax=1.5, ax=ax[g_ind])


# %%
