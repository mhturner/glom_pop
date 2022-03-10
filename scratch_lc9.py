from visanalysis.analysis import imaging_data, shared_analysis
import os
from glom_pop import dataio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

experiment_file_directory = dataio.get_config_file()['experiment_file_directory']


matching_series = shared_analysis.filterDataFiles(data_directory=experiment_file_directory,
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                                          'include_in_analysis': True})

target_val = 42
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=True)

# %%

voxel_erm = response_data['voxel_epoch_responses'].get(target_val)

unique_parameter_values, mean_response, _, _ = ID.getTrialAverages(voxel_erm)
voxel_concat = np.vstack(np.concatenate([mean_response[:, x, :] for x in np.arange(len(unique_parameter_values))], axis=1))

sns.clustermap(voxel_concat, col_cluster=False)


plt.plot(voxel_concat.mean(axis=0))
