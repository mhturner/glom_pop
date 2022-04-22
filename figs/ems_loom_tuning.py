import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis

from glom_pop import dataio, util


# PROTOCOL_ID = 'ExpandingMovingSpot'
PROTOCOL_ID = 'LoomingSpot'

if PROTOCOL_ID == 'ExpandingMovingSpot':
    eg_series = ('2022-04-19', 1)
    key_value = ('diameter', [2.5, 5., 10., 20., 40.])
    parameter_key = ['current_diameter', 'current_intensity', 'current_speed']
elif PROTOCOL_ID == 'LoomingSpot':
    eg_series = ('2022-04-19', 2)
    key_value = ('rv_ratio', [50., 100., 200., 400., 800., 1600.])
    parameter_key = ['current_rv_ratio', 'current_intensity']


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

# %%


def show_ems_tuning(unique_parameter_values, mean_response):
    diameters = np.unique([x[0] for x in unique_parameter_values])
    intensities = np.unique([x[1] for x in unique_parameter_values])
    speeds = np.unique([x[2] for x in unique_parameter_values])

    fh, ax = plt.subplots(len(diameters), len(speeds), figsize=(8, 8))
    [x.set_ylim([0, 1]) for x in ax.ravel()]
    [util.clean_axes(x) for x in ax.ravel()]
    for d_ind, diameter in enumerate(diameters):
        ax[d_ind, 0].set_ylabel(diameter)
        for sp_ind, speed in enumerate(speeds):
            if d_ind == 0:
                ax[0, sp_ind].set_title(speed)
            for i_ind, intensity in enumerate(intensities):
                pull_diameter_ind = np.where([diameter == x[0] for x in unique_parameter_values])[0]
                pull_speed_ind = np.where([speed == x[2] for x in unique_parameter_values])[0]
                pull_intensity_ind = np.where([intensity == x[1] for x in unique_parameter_values])[0]

                pull_ind = list(set.intersection(set(pull_diameter_ind),
                                                 set(pull_speed_ind),
                                                 set(pull_intensity_ind)))[0]

                if intensity == 0:
                    color = 'k'
                elif intensity == 1.0:
                    color = 'b'
                ax[d_ind, sp_ind].plot(mean_response[:, pull_ind, :].T, color=color)


def show_loom_tuning(unique_parameter_values, mean_response):
    rvs = np.unique([x[0] for x in unique_parameter_values])
    pull_intensity_ind = np.where([x[1] == 0 for x in unique_parameter_values])[0]

    fh, ax = plt.subplots(len(included_gloms), len(rvs), figsize=(8, 12))
    [x.set_ylim([-0.2, 1]) for x in ax.ravel()]
    [util.clean_axes(x) for x in ax.ravel()]
    for g_ind, glom in enumerate(included_gloms):
        ax[g_ind, 0].set_ylabel(glom)
        for r_ind, rv in enumerate(rvs):
            if g_ind == 0:
                ax[0, r_ind].set_title(rv)

            pull_rv_ind = np.where([rv == x[0] for x in unique_parameter_values])[0]
            pull_ind = list(set.intersection(set(pull_rv_ind),
                                             set(pull_intensity_ind)))[0]

            ax[g_ind, r_ind].plot(mean_response[g_ind, pull_ind, :], color=util.get_color_dict()[glom])
# %%

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    behavior_data = dataio.load_behavior(ID)
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    trial_avg_results = ID.getTrialAverages(epoch_response_matrix, parameter_key=parameter_key)

    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = trial_avg_results

    if PROTOCOL_ID == 'ExpandingMovingSpot':
        show_ems_tuning(unique_parameter_values, mean_response)
    elif PROTOCOL_ID == 'LoomingSpot':
        show_loom_tuning(unique_parameter_values, mean_response)







# %%
