import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from visanalysis.analysis import shared_analysis

from glom_pop import dataio, model, util

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']


# First include all gloms and all flies
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                                          'include_in_analysis': True})


# %% Full model, all gloms

ste = model.SingleTrialEncoding(data_series=matching_series, included_gloms=included_gloms)
ste.evaluate_performance(
                         model_type='LogReg',
                         iterations=100,
                         pull_eg=1,
                         classify_on_amplitude=True,
                         random_state=np.random.RandomState(seed=0)
                         )

# %% Plot some example traces
n_rows = 3
start_trial = 40
split_length = int(ste.eg_traces.shape[1] / len(included_gloms))
included_gloms
fh0, ax0 = plt.subplots(n_rows, 1, figsize=(2.5, 2))
[x.set_ylim([-3.5, 10]) for x in ax0.ravel()]
[x.set_axis_off() for x in ax0.ravel()]
for t in range(n_rows):
    for g_ind, g_name in enumerate(included_gloms):
        start_pt = g_ind * split_length
        end_pt = (g_ind+1) * split_length
        peak_point = start_pt + np.argmax(ste.eg_traces[start_trial+t, start_pt:end_pt])
        ax0[t].plot(np.arange(start_pt, end_pt), ste.eg_traces[start_trial+t, start_pt:end_pt], color=ste.colors[g_ind])
        ax0[t].plot(peak_point, ste.eg_traces[start_trial+t, peak_point], 'k.')
    print(str(ste.eg_stim_identity[start_trial+t]))

fh0.savefig(os.path.join(save_directory, 'single_trial_traces.svg'))


# %% overall performance plotting

col_names = [str(x) for x in ste.included_parameter_values]
mean_cmat = pd.DataFrame(data=ste.cmats.mean(axis=-1), index=col_names, columns=col_names)

mean_diag = np.mean(ste.performance, axis=0)
err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

# Performance per stim type
fh1, ax1 = plt.subplots(1, 1, figsize=(4, 2.5))
ax1.plot(col_names, mean_diag, 'ko')
for d_ind in range(len(col_names)):
    ax1.plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]], 'k-')

ax1.plot([0, len(ste.classifier_model.classes_)], [1/len(ste.classifier_model.classes_), 1/len(ste.classifier_model.classes_)], 'k--')
ax1.set_xticks([])
ax1.set_ylim([-0.1, 1.1])
ax1.set_ylabel('Performance')
ax1.set_xlabel('Stimulus identity')
# ax1.tick_params(axis='y', labelsize=11)
# ax1.tick_params(axis='x', labelsize=11, rotation=90)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Confusion matrix
fh2, ax2 = plt.subplots(1, 1, figsize=(3, 2.5))
sns.heatmap(mean_cmat, ax=ax2, vmin=0, vmax=1.0,
            cmap='Reds',
            cbar_kws={'label': 'Probability', 'ticks': [0, 0.5, 1.0]})

ax2.set_xlabel('Predicted stim')
ax2.set_ylabel('True stim')
ax2.set_xticks([])
ax2.set_yticks([])
print(ste.included_parameter_values)
fh1.savefig(os.path.join(save_directory, 'single_trial_performance.svg'))
fh2.savefig(os.path.join(save_directory, 'single_trial_confusion.svg'))

np.mean(ste.performance, axis=0).mean()


# %% For LogReg: weight matrix
# Weights: classes x features
performance_weighted_beta = (ste.classifier_model.coef_.T * mean_diag)

# weights = pd.DataFrame(data=ste.classifier_model.coef_.T,
#                        index=included_gloms)
weights = pd.DataFrame(data=performance_weighted_beta,
                       index=included_gloms)

lim = np.max(np.abs(weights.to_numpy()))
fh4, ax4 = plt.subplots(1, 1, figsize=(4, 3))
sns.heatmap(weights, ax=ax4,
            cmap='RdBu', vmax=lim, vmin=-lim,
            cbar_kws={'label': 'Performance weighted coef'})
# ax4.set_xticklabels(ste.included_parameter_values, rotation=90)
ax4.set_xticklabels([])
ax4.set_xticks([])

fh4.savefig(os.path.join(save_directory, 'single_trial_weights.svg'))

# %% Measure performance with subsets of gloms
clusters = [['LC11', 'LC21', 'LC18'],
            ['LC6', 'LC26', 'LC16', 'LPLC2'],
            ['LC4', 'LPLC1', 'LC9'],
            ['LC17', 'LC12', 'LC15']]

cluster_performance = []
for clust_ind, included_gloms in enumerate(clusters):
    ste_clust = model.SingleTrialEncoding(data_series=matching_series, included_gloms=included_gloms)
    ste_clust.evaluate_performance(
                                    model_type='LogReg',
                                    iterations=100,
                                    pull_eg=1,
                                    classify_on_amplitude=True,
                                    random_state=np.random.RandomState(seed=0)
                                    )

    cluster_performance.append(ste_clust.performance)


# %% Plot
fh5, ax5 = plt.subplots(4, 1, figsize=(4.0, 5.5))
ax5 = ax5.ravel()
[x.set_xticks([]) for x in ax5]
[x.set_ylim([-0.1, 1.1]) for x in ax5]
[x.spines['top'].set_visible(False) for x in ax5]
[x.spines['right'].set_visible(False) for x in ax5]

col_names = [str(x) for x in ste.included_parameter_values]
mean_cmat = pd.DataFrame(data=ste.cmats.mean(axis=-1), index=col_names, columns=col_names)

# Plot performance for each cluster:
colors = ['b',
          'g',
          'y',
          'm']
offset = [-0.2, 0, 0.2]
for clust_ind in range(4):
    # Chance line
    ax5[clust_ind].axhline(y=1/len(ste_clust.classifier_model.classes_), color='k', zorder=10)

    # Overall performance
    mean_diag = np.mean(ste.performance, axis=0)
    err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

    ax5[clust_ind].bar(col_names, mean_diag,
                       label='All glomeruli' if clust_ind==0 else '_',
                       color=[0.5, 0.5, 0.5])
    for d_ind in range(len(col_names)):
        ax5[clust_ind].plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
                            color=[0.5, 0.5, 0.5])

    # Single group performance
    mean_diag = np.mean(cluster_performance[clust_ind], axis=0)
    err_diag = np.std(cluster_performance[clust_ind], axis=0) / np.sqrt(cluster_performance[clust_ind].shape[0])

    ax5[clust_ind].bar(np.arange(0, len(mean_diag)), mean_diag,
                       color=colors[clust_ind],
                       label='Group {}'.format(clust_ind+1))
    for d_ind in range(len(col_names)):
        ax5[clust_ind].plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
                            linestyle='-',
                            color=colors[clust_ind])

ax5[3].set_xlabel('Stimulus identity')
fh5.supylabel('Performance')
fh5.savefig(os.path.join(save_directory, 'single_trial_cluster_performance.svg'))


# %% Compare correlated vs shuffled

# Go back to using all gloms
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# FULL MODEL, UNSHUFFLED
ste = model.SingleTrialEncoding(data_series=matching_series, included_gloms=included_gloms)
ste.evaluate_performance(
                         model_type='LogReg',
                         iterations=100,
                         pull_eg=1,
                         classify_on_amplitude=True,
                         random_state=np.random.RandomState(seed=0),
                         shuffle_trials=False
                         )

# SHUFFLED TRIALS
ste_shuffled = model.SingleTrialEncoding(data_series=matching_series, included_gloms=included_gloms)
ste_shuffled.evaluate_performance(
                         model_type='LogReg',
                         iterations=100,
                         pull_eg=1,
                         classify_on_amplitude=True,
                         random_state=np.random.RandomState(seed=0),
                         shuffle_trials=True
                         )
#  %% Shuffle analysis, plotting...

np.mean(ste.performance, axis=1).mean()
np.mean(ste_shuffled.performance, axis=1).mean()

(np.mean(ste.performance, axis=1).mean() - np.mean(ste_shuffled.performance, axis=1).mean()) / np.mean(ste.performance, axis=1).mean()

# Overall performance (mean across all stims), for each fly
fh0, ax0 = plt.subplots(1, 1, figsize=(2, 2))
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
fly_mean_intact = np.mean(ste.performance, axis=1)  # mean across all stims. shape = n flies
fly_mean_shuffle = np.mean(ste_shuffled.performance, axis=1)
ax0.plot(fly_mean_intact, fly_mean_shuffle, 'k.', alpha=0.5) # each pt is fly
ax0.errorbar(x=np.mean(fly_mean_intact), y=np.mean(fly_mean_shuffle),
             xerr=np.std(fly_mean_intact)/np.sqrt(fly_mean_intact.shape[0]),
             yerr=np.std(fly_mean_shuffle)/np.sqrt(fly_mean_shuffle.shape[0]), color='k', marker='o')
ax0.plot([0, 0.55], [0, 0.55], 'k--')
ax0.set_xlabel('Performance, unshuffled')
ax0.set_ylabel('Performance, shuffled')

fh0.savefig(os.path.join(save_directory, 'single_trial_shuffle_flies.svg'))
# %%


# Performance, by stim type
col_names = [str(x) for x in ste.included_parameter_values]
mean_intact = np.mean(ste.performance, axis=0)
err_intact = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

mean_shuffle= np.mean(ste_shuffled.performance, axis=0)
err_shuffle = np.std(ste_shuffled.performance, axis=0) / np.sqrt(ste_shuffled.performance.shape[0])

# Performance per stim type
fh1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
ax1.bar(np.arange(0, len(mean_intact)), mean_intact,
        color=[0.5, 0.5, 0.5], label='Unshuffled')
for d_ind in range(len(col_names)):
    ax1.plot([d_ind, d_ind], [mean_intact[d_ind]-err_intact[d_ind], mean_intact[d_ind]+err_intact[d_ind]],
             color=[0.5, 0.5, 0.5], linestyle='-', linewidth=2)

ax1.bar(np.arange(0, len(mean_shuffle)), mean_shuffle,
        color=[0.75, 0.2, 0], label='Shuffled')
for d_ind in range(len(col_names)):
    ax1.plot([d_ind, d_ind], [mean_shuffle[d_ind]-err_shuffle[d_ind], mean_shuffle[d_ind]+err_shuffle[d_ind]],
             color=[0.75, 0.2, 0], linestyle='-', linewidth=2)

ax1.axhline(y=1/len(ste_clust.classifier_model.classes_), color='k', zorder=10)
ax1.set_xticks([])
ax1.set_ylim([-0.1, 1.1])
ax1.set_ylabel('Performance')
ax1.set_xlabel('Stimulus identity')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fh1.legend()

fh1.savefig(os.path.join(save_directory, 'single_trial_shuffle_stims.svg'))

# %%

# %% SUPP: DECODE USING FULL RESPONSE TRACES

clusters = [['LC11', 'LC21', 'LC18'],
            ['LC6', 'LC26', 'LC16', 'LPLC2'],
            ['LC4', 'LPLC1', 'LC9'],
            ['LC17', 'LC12', 'LC15']]

cluster_performance = []
for clust_ind, included_gloms in enumerate(clusters):
    ste_clust = model.SingleTrialEncoding(data_series=matching_series, included_gloms=included_gloms)
    ste_clust.evaluate_performance(
                                    model_type='LogReg',
                                    iterations=100,
                                    pull_eg=1,
                                    classify_on_amplitude=False,
                                    random_state=np.random.RandomState(seed=0)
                                    )

    cluster_performance.append(ste_clust.performance)


# %% Plot
fh5, ax5 = plt.subplots(4, 1, figsize=(4.0, 5.5))
ax5 = ax5.ravel()
[x.set_xticks([]) for x in ax5]
[x.set_ylim([-0.1, 1.1]) for x in ax5]
[x.spines['top'].set_visible(False) for x in ax5]
[x.spines['right'].set_visible(False) for x in ax5]

col_names = [str(x) for x in ste.included_parameter_values]
mean_cmat = pd.DataFrame(data=ste.cmats.mean(axis=-1), index=col_names, columns=col_names)

# Plot performance for each cluster:
colors = ['b',
          'g',
          'y',
          'm']
offset = [-0.2, 0, 0.2]
for clust_ind in range(4):
    # Chance line
    ax5[clust_ind].axhline(y=1/len(ste_clust.classifier_model.classes_), color='k', zorder=10)

    # Overall performance
    mean_diag = np.mean(ste.performance, axis=0)
    err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

    ax5[clust_ind].bar(col_names, mean_diag,
                       label='All glomeruli' if clust_ind==0 else '_',
                       color=[0.5, 0.5, 0.5])
    for d_ind in range(len(col_names)):
        ax5[clust_ind].plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
                            color=[0.5, 0.5, 0.5])

    # Single group performance
    mean_diag = np.mean(cluster_performance[clust_ind], axis=0)
    err_diag = np.std(cluster_performance[clust_ind], axis=0) / np.sqrt(cluster_performance[clust_ind].shape[0])

    ax5[clust_ind].bar(np.arange(0, len(mean_diag)), mean_diag,
                       color=colors[clust_ind],
                       label='Group {}'.format(clust_ind+1))
    for d_ind in range(len(col_names)):
        ax5[clust_ind].plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
                            linestyle='-',
                            color=colors[clust_ind])

ax5[3].set_xlabel('Stimulus identity')
fh5.supylabel('Performance')
# fh5.savefig(os.path.join(save_directory, 'supp_ste_cluster_traces_performance.svg'))
