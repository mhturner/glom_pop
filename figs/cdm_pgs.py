from visanalysis.analysis import imaging_data, shared_analysis

import matplotlib.pyplot as plt
import numpy as np
import os

from glom_pop import dataio, util

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PGS_Reduced',
                                                                          'include_in_analysis': True,
                                                                          'num_epochs': 60})

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

for f_ind, fly in enumerate(np.unique([x['anatomical_brain'] for x in matching_series])):
    condition_tags = ['PRE', 'DRUG', 'WASH 2']
    fh, ax = plt.subplots(len(condition_tags), len(included_gloms), figsize=(9, 3))
    [x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
    [util.clean_axes(x) for x in ax.ravel()]

    fh1, ax1 = plt.subplots(2, len(included_gloms), figsize=(9, 3))
    [x.set_ylim([-0.15, 1.0]) for x in ax1.ravel()]
    [util.clean_axes(x) for x in ax1.ravel()]

    for ct_ind, condition_tag in enumerate(condition_tags):
        series = [x for x in matching_series if condition_tag in x['series_notes'] and x['anatomical_brain'] == fly][0]
        ID = imaging_data.ImagingDataObject(series['file_name'] + '.hdf5',
                                            series['series'],
                                            quiet=True)
        response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

        epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
        unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)
        concat_response = np.concatenate([mean_response[:, x, :] for x in range(mean_response.shape[1])], axis=1)
        ax[ct_ind, 0].set_ylabel(condition_tag.split(' ')[0])
        for g_ind, glom in enumerate(included_gloms):
            ax[0, g_ind].set_title(glom)
            ax[ct_ind, g_ind].plot(concat_response[g_ind, :], color=util.get_color_dict()[glom])


        # Effect of behavior on selected stim
        if condition_tag == 'DRUG':
            # Get behavior data
            behavior_data = dataio.load_behavior(ID, process_behavior=True)
            behaving_trials = np.where(behavior_data.get('behaving'))[0]
            nonbehaving_trials = np.where(~behavior_data.get('behaving'))[0]
            behaving_trials
            nonbehaving_trials

            erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                       ID,
                                                                       query={'current_diameter': 15.0},
                                                                       return_inds=True)

            behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])

            if len(behaving_inds) >= 1:
                nonbehaving_responses = epoch_response_matrix[:, behaving_inds, :]
                behaving_mean = np.nanmean(nonbehaving_responses, axis=1)  # each trial average: gloms x time
                for g_ind, glom in enumerate(included_gloms):
                    ax1[0, g_ind].plot(response_data['time_vector'], behaving_mean[g_ind, :],
                                            color=util.get_color_dict()[glom], alpha=1)
                    ax1[0, g_ind].plot(response_data['time_vector'], nonbehaving_responses[g_ind, :].T,
                                            color=util.get_color_dict()[glom], alpha=1)

            nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
            if len(nonbehaving_inds) >= 1:
                nonbehaving_responses = epoch_response_matrix[:, nonbehaving_inds, :]
                nonbehaving_mean = np.nanmean(nonbehaving_responses, axis=1)  # each trial average: gloms x time
                for g_ind, glom in enumerate(included_gloms):
                    # ax1[1, g_ind].plot(response_data['time_vector'], nonbehaving_mean[g_ind, :],
                    #                         color=util.get_color_dict()[glom], alpha=1.0)
                    ax1[1, g_ind].plot(response_data['time_vector'], nonbehaving_responses[g_ind, :].T,
                                            color=util.get_color_dict()[glom], alpha=1.0)


 # %%
 unique_parameter_values



# %%
