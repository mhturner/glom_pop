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
base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'
path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'


vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %% Include all gloms and all flies
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.getIncludedGloms(path_to_yaml)
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.getGlomValsFromNames(included_gloms)

dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

ste = model.SingleTrialEncoding(dataset=dataset, included_vals=included_vals)
ste.evaluatePerformance(
                        model_type='LogReg',
                        # model_type='RandomForest',
                        iterations=20,
                        pull_eg=1,
                        classify_on_amplitude=True,
                        )


# %%

# %% Plot some example traces
n_rows = 3
start_trial = 0

split_length = int(ste.eg_traces.shape[1] / len(included_gloms))
included_gloms
fh0, ax0 = plt.subplots(n_rows, 1, figsize=(6, 3))
[x.set_ylim([-3.5, 10]) for x in ax0.ravel()]
[x.set_axis_off() for x in ax0.ravel()]
for t in range(n_rows):
    for g_ind, g_name in enumerate(included_gloms):
        start_pt = g_ind * split_length
        end_pt = (g_ind+1) * split_length
        ax0[t].plot(np.arange(start_pt, end_pt), ste.eg_traces[start_trial+t, start_pt:end_pt], color=ste.colors[g_ind, :])
    ax0[t].set_title(str(ste.eg_stim_identity[start_trial+t]))

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
fh2, ax2 = plt.subplots(1, 1, figsize=(3.0, 2.5))
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


# %% For LogReg: weight matrix
# Weights: classes x features
performance_weighted_beta = (ste.classifier_model.coef_.T * mean_diag)

# weights = pd.DataFrame(data=ste.classifier_model.coef_,
#                        index=ste.included_parameter_values,
#                        columns=included_gloms)
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
