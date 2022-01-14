import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from seaborn import desaturate

from glom_pop import dataio, model


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'
path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

# %% Include all gloms and all flies

included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

ste = model.SingleTrialEncoding()
ste.evaluatePerformance(dataset=dataset,
                        included_gloms=included_gloms,
                        iterations=20,
                        pull_eg=0,
                        model_type='RandomForest',
                        # model_type='LogReg',
                        )

# %% Plot example traces
# TODO: track down very noisy single trials here. Is this because of dF/F calc and bad baseline?
n_rows = 4
split_length = int(ste.X_test.shape[1] / len(included_gloms))
included_gloms
fh0, ax0 = plt.subplots(n_rows, 1, figsize=(6, 4))
# [x.set_ylim([-0.5, 0.8]) for x in ax0.ravel()]
[x.set_axis_off() for x in ax0.ravel()]
for t in range(n_rows):
    for g_ind, g_name in enumerate(included_gloms):
        start_pt = g_ind * split_length
        end_pt = (g_ind+1) * split_length
        ax0[t].plot(np.arange(start_pt, end_pt), ste.X_test[t, start_pt:end_pt], color=ste.colors[g_ind, :])
    ax0[t].set_title(str(ste.unique_parameter_values[ste.y_test[t]]))

fh0.savefig(os.path.join(save_directory, 'single_trial_traces.svg'))


# %% overall performance plotting

col_names = [str(x) for x in ste.included_parameter_values]
mean_cmat = pd.DataFrame(data=ste.cmats.mean(axis=-1), index=col_names, columns=col_names)

mean_diag = np.mean(ste.performance, axis=0)
err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

# Performance per stim type
fh1, ax1 = plt.subplots(1, 1, figsize=(6, 3))
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
fh2, ax2 = plt.subplots(1, 1, figsize=(3.5, 2.75))
sns.heatmap(mean_cmat, ax=ax2, vmin=0, vmax=1.0, cmap='Reds', cbar_kws={'label': 'Probability'})
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_xticks([])
ax2.set_yticks([])
print(ste.included_parameter_values)
fh1.savefig(os.path.join(save_directory, 'single_trial_performance.svg'))
fh2.savefig(os.path.join(save_directory, 'single_trial_confusion.svg'))


# %% logistic regression cartoon schematics

xx = np.linspace(-20, 20, 100)
yy = 1 / (1+np.exp(-xx))

fh3, ax3 = plt.subplots(1, 1, figsize=(2, 1.5))
ax3.plot(xx, yy, color='k', linewidth=2)
ax3.set_xlabel('$\hat{y}$', fontsize=14)
ax3.set_ylabel('p', fontsize=14)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

fh3.savefig(os.path.join(save_directory, 'single_trial_log_cartoon.png'), bbox_inches='tight')


# %%
# Weights: classes x features
weights = ste.LogRegModel.coef_


sns.heatmap(weights)

print('min = {:.2f} / max = {:.2f}'.format(weights.min(), weights.max()))

fh4, ax4 = plt.subplots(len(included_gloms), 1, figsize=(4, 8))
for g_ind, g_name in enumerate(included_gloms):
    start_pt = g_ind * split_length
    end_pt = (g_ind+1) * split_length
    wt = weights[:, start_pt:end_pt]
    ax4[g_ind].plot(np.arange(0, wt.shape[0]), np.max(np.abs(wt), axis=-1),
                    linestyle='none', marker='o', color=ste.colors[g_ind, :])
    ax4[g_ind].plot([0, wt.shape[0]], [0, 0], 'k--')
    ax4[g_ind].set_axis_off()
    # ax4[g_ind].set_ylim([0, 1])


# %% Measure performance with subsets of gloms

fh5, ax5 = plt.subplots(1, 1, figsize=(6, 3))
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_xticks([])
ax5.set_ylim([-0.1, 1.1])

xoffset = 0.1

dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

# First cluster:
included_gloms = ['LC26', 'LC6', 'LC12', 'LC15']
ste = model.SingleTrialEncoding()
ste.evaluatePerformance(dataset=dataset,
                        included_gloms=included_gloms,
                        iterations=20,
                        pull_eg=0)

col_names = [str(x) for x in ste.included_parameter_values]
mean_diag = np.mean(ste.performance, axis=0)
err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

ax5.plot([0, len(ste.LogRegModel.classes_)], [1/len(ste.LogRegModel.classes_), 1/len(ste.LogRegModel.classes_)], 'k--')
ax5.plot(np.arange(0, len(col_names))-xoffset, mean_diag, marker='o', color=desaturate('r', 0.6), linestyle='None', alpha=0.5)
for d_ind in range(len(col_names)):
    ax5.plot([d_ind-xoffset, d_ind-xoffset], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
             color=desaturate('r', 0.6), linestyle='-', marker='None', alpha=0.5)


# Second cluster:
included_gloms = ['LC9', 'LC11', 'LC18', 'LC21']
ste = model.SingleTrialEncoding()
ste.evaluatePerformance(dataset=dataset,
                        included_gloms=included_gloms,
                        iterations=20,
                        pull_eg=0)

mean_diag = np.mean(ste.performance, axis=0)
err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

ax5.plot(np.arange(0, len(col_names)), mean_diag, marker='o', color=desaturate('g', 0.6), linestyle='None', alpha=0.5)
for d_ind in range(len(col_names)):
    ax5.plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
             color=desaturate('g', 0.6), linestyle='-', marker='None', alpha=0.5)


# Third cluster:
included_gloms = ['LPLC1', 'LC16', 'LPLC2']
ste = model.SingleTrialEncoding()
ste.evaluatePerformance(dataset=dataset,
                        included_gloms=included_gloms,
                        iterations=20,
                        pull_eg=0)

mean_diag = np.mean(ste.performance, axis=0)
err_diag = np.std(ste.performance, axis=0) / np.sqrt(ste.performance.shape[0])

ax5.plot(np.arange(0, len(col_names))+xoffset, mean_diag, marker='o', color=desaturate('b', 0.6), linestyle='None', alpha=0.5)
for d_ind in range(len(col_names)):
    ax5.plot([d_ind+xoffset, d_ind+xoffset], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]],
             color=desaturate('b', 0.6), linestyle='-', marker='None', alpha=0.5)

ax5.set_ylabel('Performance')
ax5.set_xlabel('Stimulus identity')

fh5.savefig(os.path.join(save_directory, 'single_trial_clust_performance.svg'))

# %%
