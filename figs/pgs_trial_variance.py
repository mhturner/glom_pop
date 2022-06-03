from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from visanalysis.util import plot_tools
from sklearn.metrics import explained_variance_score
from scipy.optimize import least_squares

from glom_pop import dataio, util, model


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
                                                                          'num_epochs': 180},
                                                  target_groups=['aligned_response'])

eg_series = ('2022-03-01', 2)

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# %%
all_cmats = []
r_gain_behavior = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
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
        gain_for_stim = response_amplitude[:, pind] / np.nanmean(response_amplitude[:, pind], axis=1)[:, np.newaxis]
        gain_by_trial[:, up, :] = gain_for_stim
        new_cmat = pd.DataFrame(data=gain_for_stim.T).corr().to_numpy()  # glom x glom. Corr map for this stim type
        cmats.append(new_cmat)

    all_cmats.append(np.stack(cmats, axis=-1))

    if dataio.has_behavior_data(ID):
        print('HAS BEHAVIOR')
        behavior_data = dataio.load_behavior(ID, process_behavior=True)

        r_gain_behavior_new = np.zeros((len(included_gloms), len(pull_inds)))
        r_gain_behavior_new[:] = np.nan
        for up, pind in enumerate(pull_inds):
            beh = behavior_data['running_amp'][0, pind]  # for all trials of this stim type
            for g_ind, glom in enumerate(included_gloms):
                gn = gain_by_trial[g_ind, up, :]
                r = np.corrcoef(beh, gn)[0, 1]
                r_gain_behavior_new[g_ind, up] = r

        r_gain_behavior.append(r_gain_behavior_new)

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
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

    print('--------')
all_cmats = np.stack(all_cmats, axis=-1)  # (glom x glom x stim x fly)
r_gain_behavior = np.stack(r_gain_behavior, axis=-1)  # (gloms x stim x fly)
fh0.savefig(os.path.join(save_directory, 'pgs_variance_eg_fly.svg'), transparent=True)


# %% glom-glom correlation matrix, across all stims
fly_average_cmats = np.nanmean(all_cmats, axis=-1)  # shape = glom x glom x stim

glom_corr_df = pd.DataFrame(data=np.nanmean(fly_average_cmats, axis=-1), index=included_gloms, columns=included_gloms)

fh1, ax1 = plt.subplots(1, 1, figsize=(3, 2.5))
sns.heatmap(glom_corr_df,
            ax=ax1,
            vmin=0, vmax=+1,
            cmap='Reds',
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Gain-gain corr. (r)'}
            )
fh1.savefig(os.path.join(save_directory, 'pgs_variance_corrmat.svg'), transparent=True)

# %% For flies w behavior tracking: corr between gain & behavior, for each stim and glom
unique_parameter_values
print(unique_parameter_values)
df = pd.DataFrame(data=np.nanmean(r_gain_behavior, axis=-1),
                  index=included_gloms, columns=[str(s) for s in unique_parameter_values])

fh2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
sns.heatmap(df,
            cmap='coolwarm', vmin=-0.6, vmax=+0.6,
            ax=ax2,
            yticklabels=True,
            xticklabels=False,
            cbar_kws={'label': 'Behavior-gain corr (r)'})

fh2.savefig(os.path.join(save_directory, 'pgs_variance_behcorr.svg'), transparent=True)



# %% IN PROGRESS...
# TODO: shared gain model
# TODO: corr with behavior
eg_ind = 13

series = matching_series[eg_ind]
series_number = series['series']
file_path = series['file_name'] + '.hdf5'
file_name = os.path.split(series['file_name'])[-1]
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)


# %% shared gain model

r_vals = []
for s_ind, series in enumerate(matching_series):

    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    behavior_data = dataio.load_behavior(ID, process_behavior=True)
    K = 4

    if np.any(np.isnan(epoch_response_matrix)):
        pass
    else:
        # TODO: throws an error with nans. If nans, toss that glom and do the rest...
        mod = model.SharedGainModel(ID, epoch_response_matrix)
        mod.fit_model(K=K)
        res = mod.evaluate_performance()

    # plot model results...

        fh0, ax0 = plt.subplots(1, 1, figsize=(4, 4))
        sns.heatmap(pd.DataFrame(data=mod.W_fit, index=included_gloms), cmap='RdBu_r', vmin=-1, vmax=1)

        fh1, ax1 = plt.subplots(K, 1, figsize=(8, 4))
        new_r = []
        for k in range(K):
            ax1[k].plot(mod.M_fit[k, :])
            twax = ax1[k].twinx()
            r = np.corrcoef(mod.M_fit[k, :], behavior_data['running_amp'][0, :])[0, 1]
            new_r.append(r)
            twax.plot(behavior_data['running_amp'][0, :], 'k')
            twax.set_title('r = {:.2f}'.format(r))

        r_vals.append(new_r)

        fh0.savefig(os.path.join(save_directory, 'lsq_model_heatmap_{}.svg'.format(s_ind)), transparent=True)
        fh1.savefig(os.path.join(save_directory, 'lsq_model_{}.svg'.format(s_ind)), transparent=True)

r_vals = np.vstack(r_vals)
# %%
fh2, ax2 = plt.subplots(1, 1, figsize=(8, 3))
ax2.plot(r_vals, linestyle=None, marker='o', color='k')

# %% TODO: cycle thru K
# TODO: cross-validation? how to do this for this model...
mod = model.SharedGainModel(ID, epoch_response_matrix)
r2_vals = []
for K in range(1, 5):
    mod.fit_model(K=K)
    res = mod.evaluate_performance()
    r2_vals.append(res.get('r2'))
r2_vals = np.vstack(r2_vals)


# %%

fh, ax = plt.subplots(1, 1, figsize=(3, 3))
for g_ind, glom in enumerate(included_gloms):
    ax.plot(np.arange(1, 5), r2_vals[:, g_ind], marker='o', color=util.get_color_dict()[glom])

# %%





# %%
