from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from glom_pop import dataio, util, model
import pandas as pd


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PGS_Reduced',
                                                                          'include_in_analysis': True,
                                                                          'num_epochs': 180},
                                                  target_groups=['aligned_response'])

n_trials = 180
frac_train = 0.90
iterations = 20

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))

included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# %%
performance = []
performance_behaving = []
performance_nonbehaving = []

confusion_matrix = []
confusion_matrix_behaving = []
confusion_matrix_nonbehaving = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    if ft_data_path:  # Only when has behavior
        behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                                 response_len=response_data.get('response').shape[1],
                                                 process_behavior=True, fps=50, exclude_thresh=300, show_qc=True)

        behaving_trials = np.where(behavior_data.get('is_behaving')[0])[0]
        nonbehaving_trials = np.where(~behavior_data.get('is_behaving')[0])[0]
        if len(behaving_trials) < 2:
            continue

        # Single trial decoding model
        ste = model.SingleTrialEncoding_onefly(ID, included_gloms)
        ste.prep_model()

        p_all = []
        cmats_all = []
        p_beh = []
        cmats_beh = []
        p_nonbeh = []
        cmats_nonbeh = []
        for it in range(iterations):
            # train on all trials
            # train_inds = np.random.choice(np.arange(n_trials), int((frac_train * n_trials)))

            # (1) Test on all trials:
            train_inds = np.random.choice(np.arange(n_trials), int((frac_train * n_trials)))
            test_inds = np.array([x for x in np.arange(n_trials) if x not in train_inds])
            ste.train_test_model(train_inds, test_inds)
            p_all.append(ste.performance)
            cmats_all.append(ste.cmat)

            # (2) Test on behaving trials
            train_inds = np.random.choice(behaving_trials, int((frac_train * len(behaving_trials))))
            test_beh = np.array([x for x in behaving_trials if x not in train_inds])
            if len(test_beh) > 0:
                ste.train_test_model(train_inds, test_beh)
                if ste.cmat.shape[0] == 6:  # make sure each stim is actually sampled
                    p_beh.append(ste.performance)
                    cmats_beh.append(ste.cmat)

            # (3) Test on nonbehaving trials
            train_inds = np.random.choice(nonbehaving_trials, int((frac_train * len(nonbehaving_trials))))
            test_nonbeh = np.array([x for x in nonbehaving_trials if x not in train_inds])
            if len(test_nonbeh) > 0:
                ste.train_test_model(train_inds, test_nonbeh)
                if ste.cmat.shape[0] == 6:
                    p_nonbeh.append(ste.performance)
                    cmats_nonbeh.append(ste.cmat)
        performance.append(np.mean(np.stack(p_all, axis=-1), axis=-1))
        confusion_matrix.append(np.mean(np.stack(cmats_all, axis=-1), axis=-1))

        performance_behaving.append(np.mean(np.stack(p_beh, axis=-1), axis=-1))
        confusion_matrix_behaving.append(np.mean(np.stack(cmats_beh, axis=-1), axis=-1))

        performance_nonbehaving.append(np.mean(np.stack(p_nonbeh, axis=-1), axis=-1))
        confusion_matrix_nonbehaving.append(np.mean(np.stack(cmats_nonbeh, axis=-1), axis=-1))

p_by_stim = np.vstack([np.diag(x) for x in confusion_matrix])  # fly x stim
p_by_stim_beh = np.vstack([np.diag(x) for x in confusion_matrix_behaving])
p_by_stim_nonbeh = np.vstack([np.diag(x) for x in confusion_matrix_nonbehaving])

ste.unique_parameter_values
# %%

fh, ax = plt.subplots(1, 3, figsize=(9, 3))
sns.heatmap(np.mean(np.stack(confusion_matrix, axis=-1), axis=-1), vmin=0, vmax=1, ax=ax[0])

sns.heatmap(np.mean(np.stack(confusion_matrix_behaving, axis=-1), axis=-1), vmin=0, vmax=1, ax=ax[1])

sns.heatmap(np.mean(np.stack(confusion_matrix_nonbehaving, axis=-1), axis=-1), vmin=0, vmax=1, ax=ax[2])


# %%
fh, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
mean_overall = np.mean(p_by_stim, axis=0)
err_overall = np.std(p_by_stim, axis=0) / np.sqrt(p_by_stim.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_overall,
            yerr=err_overall,
            color='k', marker='o', linestyle='None', label='All trials')

mean_beh = np.mean(p_by_stim_beh, axis=0)
err_beh = np.std(p_by_stim_beh, axis=0) / np.sqrt(p_by_stim_beh.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_beh,
            yerr=err_beh,
            color='g', marker='o', linestyle='None', label='Walking')

mean_nonbeh = np.mean(p_by_stim_nonbeh, axis=0)
err_nonbeh = np.std(p_by_stim_nonbeh, axis=0) / np.sqrt(p_by_stim_nonbeh.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_nonbeh,
            yerr=err_nonbeh,
            color='r', marker='o', linestyle='None', label='Stationary')

ax.set_ylim([0, 1])
ax.axhline(y=1/6, color=[0.5, 0.5, 0.5])
ax.set_ylabel('Performace')
fh.legend()


# %%
all_responses = []

trial_amps = []
stim_codes = []
behaving_codes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    if ft_data_path:  # Only when has behavior
        behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                                 response_len=response_data.get('response').shape[1],
                                                 process_behavior=True, fps=50, exclude_thresh=300, show_qc=True)

        behaving_trials = np.where(behavior_data.get('is_behaving')[0])[0]
        nonbehaving_trials = np.where(~behavior_data.get('is_behaving')[0])[0]

        parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
        unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
        epoch_inds = [np.where([pv == up for pv in parameter_values])[0] for up in unique_parameter_values]

        # gloms x trials
        df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])
        trial_amps.append(ID.getResponseAmplitude(epoch_response_matrix, metric='max'))
        stim_codes.append(df['params'].apply(lambda x: list(unique_parameter_values).index(x)).values)
        behaving_codes.append(behavior_data.get('is_behaving')[0])

        trial_averages = np.zeros((len(included_gloms), len(unique_parameter_values), 2, epoch_response_matrix.shape[-1]))  # glom x stim x beh/nonbeh x time
        trial_averages[:] = np.nan
        for u_ind, up in enumerate(unique_parameter_values):
            pull_beh_inds = [x for x in epoch_inds[u_ind] if x in behaving_trials]
            pull_nonbeh_inds = [x for x in epoch_inds[u_ind] if x in nonbehaving_trials]

            trial_averages[:, u_ind, 0, :] = np.nanmean(epoch_response_matrix[:, pull_beh_inds, :], axis=1)
            trial_averages[:, u_ind, 1, :] = np.nanmean(epoch_response_matrix[:, pull_nonbeh_inds, :], axis=1)

        all_responses.append(trial_averages)

all_responses = np.stack(all_responses, axis=-1)  # glom x stim x beh/nonbeh x time x fly


# %%
fh, ax = plt.subplots(len(included_gloms), 6, figsize=(8, 8))
[x.set_ylim([-0.1, 0.5]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
for u_ind, up in enumerate(unique_parameter_values):
    for g_ind, glom in enumerate(included_gloms):
        ax[g_ind, u_ind].plot(np.nanmean(all_responses[g_ind, u_ind, 0, :, :], axis=-1), color=util.get_color_dict()[glom])
        ax[g_ind, u_ind].plot(np.nanmean(all_responses[g_ind, u_ind, 1, :, :], axis=-1), color=util.get_color_dict()[glom], alpha=0.5)


# %% balance of pop'n responses, behaving vs. nonbehaving
# z score each individual glom, across all conditions and stims

all_response_amps = np.stack([ID.getResponseAmplitude(all_responses[..., x], metric='max') for x in range(all_responses.shape[-1])], axis=-1)
mean_sub = all_response_amps - np.nanmean(all_response_amps, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
scored = mean_sub / np.nanstd(all_response_amps, axis=(1, 2))[:, np.newaxis, np.newaxis, :]  # glom, stim, beh, fly
fly_avg_scored = np.nanmean(scored, axis=-1)  # shape = gloms x stims x beh
fly_err_scored = np.nanstd(scored, axis=-1) / np.sqrt(scored.shape[-1])
# %%
fh, ax = plt.subplots(6, 1, figsize=(5, 10), tight_layout=True)
beh_gloms = np.mean(fly_avg_scored[:, :, 0], axis=1)
nonbeh_gloms = np.mean(fly_avg_scored[:, :, 1], axis=1)
[x.set_ylim([-2, 2.5]) for x in ax]
[x.spines['right'].set_visible(False) for x in ax]
[x.spines['top'].set_visible(False) for x in ax]
for s in range(6):
    ax[s].set_title(unique_parameter_values[s][0])
    sort_inds = np.argsort(fly_avg_scored[:, s, 1])[::-1]
    ax[s].axhline(y=0, color=[0.5, 0.5, 0.5], alpha=0.5)
    ax[s].errorbar(x=np.arange(len(included_gloms)),
                   y=fly_avg_scored[:, s, 0][sort_inds],
                   yerr=fly_err_scored[:, s, 0][sort_inds],
                   color='g', marker='o')

    ax[s].errorbar(x=np.arange(len(included_gloms)),
                   y=fly_avg_scored[:, s, 1][sort_inds],
                   yerr=fly_err_scored[:, s, 1][sort_inds],
                   color='r', marker='o')

    ax[s].set_xticks(np.arange(len(included_gloms)))
    ax[s].set_xticklabels(included_gloms[sort_inds], rotation=90);
fh.supylabel('Response amp (z-score)')


# %%
from sklearn.decomposition import PCA
from scipy.stats import zscore

trial_amps_z = [zscore(x, axis=1) for x in trial_amps]
X = np.hstack(trial_amps_z).T
Y_stim_codes = np.hstack(stim_codes)
Y_beh_codes = np.hstack(behaving_codes)


X[np.isnan(X)] = 0
pca = PCA(n_components=13).fit(X)
pca.explained_variance_ratio_
pc_1 = pca.components_[0, :]
pc_2 = pca.components_[1, :]

proj_1 = np.dot(pc_1, X.T)
proj_2 = np.dot(pc_2, X.T)

fh, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(proj_1, proj_2, marker='.', c=Y_stim_codes)
# ax.scatter(proj_1, proj_2, marker='.', c=Y_beh_codes)

# %%




# %%
