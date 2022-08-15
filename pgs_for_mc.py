from visanalysis.analysis import imaging_data, shared_analysis
import numpy as np
import os
import pickle
from datetime import date

from glom_pop import dataio, util


sync_dir = dataio.get_config_file()['sync_dir']
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PGS_Reduced',
                                                                          'include_in_analysis': True,
                                                                          'num_epochs': 180},
                                                  target_groups=['aligned_response'])

eg_series = ('2022-03-01', 2)

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))

included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# %%


all_fly_results = []
for s_ind, series in enumerate(matching_series):
    fly_results = {}
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    fly_results['file_name'] = os.path.split(file_path)[-1]
    fly_results['series_number'] = series_number
    fly_results['sample_period'] = ID.getAcquisitionMetadata()['sample_period']


    print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Align responses
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

    fly_results['gloms'] = included_gloms
    fly_results['epoch_response_matrix'] = epoch_response_matrix  # gloms x trials x time
    fly_results['epoch_parameter_dicts'] = ID.getEpochParameterDicts()  # len = trials


    # Load behavior data
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    if ft_data_path:

        print('HAS BEHAVIOR')
        behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                                 response_len = response_data.get('response').shape[1],
                                                 process_behavior=True, fps=50, exclude_thresh=300)

        fly_results['has_behavior_data'] = True
        fly_results['walking_amp'] = behavior_data['walking_amp'][0, :]
        fly_results['walking_response_matrix'] = behavior_data['walking_response_matrix'][0, :, :]

    else: # no behavior
        fly_results['has_behavior_data'] = False


    all_fly_results.append(fly_results)
    print('--------')

# %% Save

save_directory = os.path.join('/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/', 'data_for_mc')
fpath = os.path.join(save_directory, 'pgs_trial_variance_data_{}.pkl'.format(date.today().strftime("%Y%m%d")))
with open(fpath, 'wb') as f_handle:
    pickle.dump(all_fly_results, f_handle)
