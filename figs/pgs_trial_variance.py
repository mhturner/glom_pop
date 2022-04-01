from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from visanalysis.util import plot_tools

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
                                                                          'num_epochs': 180})

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)


# %% ALL FLIES
eg_ind = 0
all_cmats = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Align responses
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

    # Calculate gain for each trial. Gain := response amplitude normalized by median response amplitude to that stim
    parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
    pull_inds = [np.where([pv == up for pv in parameter_values])[0] for up in unique_parameter_values]
    response_amplitude = ID.getResponseAmplitude(epoch_response_matrix, metric='max')  # gloms x trials
    gain_by_trial = np.zeros((len(included_gloms),
                              len(unique_parameter_values),
                              len(pull_inds[0])))  # gloms x stim condition x trials
    gain_by_trial[:] = np.nan
    cmats = []
    for up, pind in enumerate(pull_inds):
        gain_for_stim = response_amplitude[:, pind] / np.nanmedian(response_amplitude[:, pind], axis=1)[:, np.newaxis]
        gain_by_trial[:, up, :] = gain_for_stim
        new_cmat = pd.DataFrame(data=gain_for_stim.T).corr().to_numpy()  # glom x glom. Corr map for this stim type
        cmats.append(new_cmat)

    all_cmats.append(np.stack(cmats, axis=-1))
    unique_parameter_values
    if s_ind == eg_ind:
        # Plot eg trials for eg fly
        eg_trials = np.arange(0, 5)
        fh0, ax0 = plt.subplots(len(included_gloms), 5, figsize=(6, 4), gridspec_kw={'wspace': 0.025, 'hspace':0.025})
        for up_ind, up in enumerate(np.arange(1, 6)):  # skip drifting gratings (first param set)
            concat_trial_response = np.concatenate([trial_response_by_stimulus[up][:, x, :] for x in eg_trials], axis=1)
            [x.set_ylim([-0.15, 1.1]) for x in ax0.ravel()]
            [util.clean_axes(x) for x in ax0.ravel()]
            [x.set_ylim() for x in ax0.ravel()]
            for g_ind, glom in enumerate(included_gloms):
                if up_ind == 0:
                    ax0[g_ind, up_ind].set_ylabel(glom, rotation=0)
                    if g_ind == 0:
                        plot_tools.addScaleBars(ax0[g_ind, up_ind], dT=2, dF=0.25, T_value=0, F_value=-0.1)
                if g_ind == 0:
                    trial_len = trial_response_by_stimulus[0].shape[2]
                    concat_len = concat_trial_response.shape[1]
                    y_val = 1.1
                    ax0[g_ind, up_ind].plot(np.linspace(trial_len/2,
                                                        concat_len-trial_len/2,
                                                        len(eg_trials)),
                                            y_val * np.ones(len(eg_trials)),
                                            'rv')
                ax0[g_ind, up_ind].plot(concat_trial_response[g_ind, :], color=util.get_color_dict()[glom])

all_cmats = np.stack(all_cmats, axis=-1)  # (glom x glom x stim x fly)

fh0.savefig(os.path.join(save_directory, 'pgs_variance_eg_fly.svg'), transparent=True)
# %%
fly_average_cmat = np.mean(all_cmats, axis=-1)  # shape = glom x glom x stim
fh1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
sns.heatmap(pd.DataFrame(data=fly_average_cmat.mean(axis=-1), index=included_gloms, columns=included_gloms),
            ax=ax1,
            vmin=-1, vmax=+1,
            cmap='RdBu_r',
            xticklabels=True,
            yticklabels=True)
fh1.savefig(os.path.join(save_directory, 'pgs_variance_corrmat.svg'), transparent=True)

# %%
