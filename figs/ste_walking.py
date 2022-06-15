from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from glom_pop import dataio, util, model


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
mean_overall = np.mean(p_by_stim, axis=0)
err_overall = np.std(p_by_stim, axis=0) / np.sqrt(p_by_stim.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_overall,
            yerr=err_overall,
            color='k', marker='o', linestyle='None')

mean_beh = np.mean(p_by_stim_beh, axis=0)
err_beh = np.std(p_by_stim_beh, axis=0) / np.sqrt(p_by_stim_beh.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_beh,
            yerr=err_beh,
            color='g', marker='o', linestyle='None')

mean_nonbeh = np.mean(p_by_stim_nonbeh, axis=0)
err_nonbeh = np.std(p_by_stim_nonbeh, axis=0) / np.sqrt(p_by_stim_nonbeh.shape[0])
ax.errorbar(x=np.arange(6),
            y=mean_nonbeh,
            yerr=err_nonbeh,
            color='r', marker='o', linestyle='None')

ax.set_ylim([0, 1])
ax.axhline(y=1/6, color=[0.5, 0.5, 0.5])



# %%
