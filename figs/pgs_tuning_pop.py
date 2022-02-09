from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import colorcet as cc
import pandas as pd
import ants
import seaborn as sns

from scipy.stats import zscore

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

from glom_pop import dataio, util, alignment

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'
transform_directory = os.path.join(base_dir, 'transforms', 'meanbrain_template')

# %%

# %% PLOT MEAN RESPONSES TO TUNING SUITE

glom_size_threshold = 10
path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

# Load overall glom map
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))
glom_mask_2_meanbrain = ants.image_read(os.path.join(transform_directory, 'glom_mask_reg2meanbrain.nii')).numpy()
all_vals, all_names = dataio.getGlomMaskDecoder(glom_mask_2_meanbrain)

all_sizes = pd.DataFrame(data=[np.sum(glom_mask_2_meanbrain == mv) for mv in all_vals],
                         index=all_names.values)

# print(all_sizes)

# Get included gloms from data yaml
included_gloms = dataio.getIncludedGloms(path_to_yaml)
included_vals = dataio.getGlomValsFromNames(included_gloms)

# Set colormap for included gloms
cmap = cc.cm.glasbey
colors = cmap(included_vals/included_vals.max())

# Load PGS dataset from yaml
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

all_responses = []
response_amplitudes = []
all_glom_sizes = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)

    # epoch_response_stack: shape=(gloms, time, trials)
    epoch_response_stack = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_stack[:] = np.nan

    glom_sizes = np.zeros(len(included_vals))
    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)
        glom_sizes[val_ind] = new_glom_size

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_stack[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, trial_response_by_stimulus = ID.getMeanBrainByStimulus(epoch_response_stack)
    n_stimuli = mean_voxel_response.shape[2]

    all_responses.append(mean_voxel_response)
    response_amplitudes.append(response_amp)
    all_glom_sizes.append(glom_sizes)

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)
all_glom_sizes = np.stack(all_glom_sizes, axis=-1)  # dims = (gloms, fly)

# Exclude last two stims (full field flashes)
unique_parameter_values = unique_parameter_values[:-2]
all_responses = all_responses[:, :, :-2, :]
response_amplitudes = response_amplitudes[:, :-2, :]

mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, time, param)
# Concatenate responses across stimulus types
#   Shape = (glom, concat time)
mean_concat = np.vstack(np.concatenate([mean_responses[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
sem_concat = np.vstack(np.concatenate([sem_responses[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
std_concat = np.vstack(np.concatenate([std_responses[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
time_concat = np.arange(0, mean_concat.shape[-1]) * ID.getAcquisitionMetadata().get('sample_period')

np.save(os.path.join(save_directory, 'chat_responses.npy'), all_responses)
np.save(os.path.join(save_directory, 'mean_chat_responses.npy'), mean_responses)
np.save(os.path.join(save_directory, 'sem_chat_responses.npy'), sem_responses)
np.save(os.path.join(save_directory, 'included_gloms.npy'), included_gloms)
np.save(os.path.join(save_directory, 'colors.npy'), colors)


# %% QC: Number of voxels in each glomerulus

empties = (all_glom_sizes < glom_size_threshold).sum(axis=1)


fh, ax = plt.subplots(1, 1, figsize=(9, 3))
for ind, ig in enumerate(included_gloms):
    ax.plot(ind*np.ones(all_glom_sizes.shape[1]), all_glom_sizes[ind, :], 'k.')
    ax.annotate('{}'.format(empties[ind]), (ind, 1e3))

ax.axhline(glom_size_threshold, color='r')
ax.set_xticks(np.arange(0, len(included_gloms)))
ax.set_xticklabels(included_gloms, rotation=90)
ax.set_yscale('log')

# %% QC: glom response traces. For MC or occlusion artifacts

for key in dataset:
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)

    fh, ax = plt.subplots(14, 1, figsize=(12, 12))
    ax[0].set_title(key)
    ct = 0
    for i in range(19):
        val = response_data['mask_vals'][i]
        if val in included_vals:
            ax[ct].plot(response_data['response'][i, :])
            ax[ct].set_ylabel(dataio.getGlomNameFromVal(val))
            ct += 1


# %% PLOTTING


# # # Cluster on stimulus tuning # # #
peak_responses = mean_responses.max(axis=1)  # (glom x stim)
peak_responses = zscore(peak_responses, axis=1)

# Compute linkage matrix, Z
Z = linkage(peak_responses,
            method='average',
            metric='euclidean',
            optimal_ordering=True)

# DENDROGRAM
fh0, ax0 = plt.subplots(1, 1, figsize=(1.5, 6))
D = dendrogram(Z, p=len(included_gloms), truncate_mode='lastp', ax=ax0,
               above_threshold_color='black',
               color_threshold=1, orientation='left',
               labels=included_gloms)
leaves = leaves_list(Z)  # glom indices

# ax0.set_yticks([])
ax0.set_xticks([])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.invert_yaxis()

# Plot mean concatenated responses
fh1, ax1 = plt.subplots(len(included_gloms), 1, figsize=(12, 6))
[util.cleanAxes(x) for x in ax1.ravel()]


fh1.subplots_adjust(wspace=0.00, hspace=0.00)
for u_ind, un in enumerate(unique_parameter_values):
    for leaf_ind, g_ind in enumerate(leaves):
        name = included_gloms[g_ind]
        # ax1[leaf_ind].set_ylabel(name, fontsize=11, rotation=0)
        # ax1[leaf_ind].fill_between(time_concat,
        #                            mean_concat[g_ind, :]-sem_concat[g_ind, :],
        #                            mean_concat[g_ind, :]+sem_concat[g_ind, :],
        #                            facecolor=sns.desaturate(colors[g_ind, :], 0.25), alpha=1.0, linewidth=0,
        #                            rasterized=True)
        ax1[leaf_ind].plot(time_concat, mean_concat[g_ind, :],
                           color=colors[g_ind, :], alpha=1.0, linewidth=1.0,
                           rasterized=True)
        if (leaf_ind == 0):
            plot_tools.addScaleBars(ax1[leaf_ind], dT=2, dF=0.25, T_value=0, F_value=-0.1)

[x.set_ylim([mean_responses.min(), 1.1*mean_responses.max()]) for x in ax1.ravel()]

fh0.savefig(os.path.join(save_directory, 'pgs_tuning_dendrogram.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'pgs_mean_tuning.svg'), transparent=True, dpi=300)

# Save leaves list and ordered response dataframe
np.save(os.path.join(save_directory, 'cluster_leaves_list.npy'), leaves)
concat_df = pd.DataFrame(data=mean_concat[leaves, :], index=np.array(included_gloms)[leaves])
concat_df.to_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl'))



# %% glom highlight maps

# Load mask key for VPN types

glom_mask_2_meanbrain = alignment.filterGlomMask_by_name(mask=glom_mask_2_meanbrain,
                                                         vpn_types=vpn_types,
                                                         included_gloms=included_gloms)

fh6, ax6 = plt.subplots(len(leaves), 1, figsize=(2, 6))
[x.set_xlim([30, 230]) for x in ax6.ravel()]
[x.set_ylim([180, 5]) for x in ax6.ravel()]
[x.set_axis_off() for x in ax6.ravel()]


for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    glom_mask_val = vpn_types.loc[vpn_types.get('vpn_types') == name, 'Unnamed: 0'].values[0]

    util.makeGlomMap(ax=ax6[leaf_ind],
                     glom_map=glom_mask_2_meanbrain,
                     z_val=None,
                     highlight_vals=[glom_mask_val])

fh6.savefig(os.path.join(save_directory, 'pgs_glom_highlights.svg'), transparent=True)
# %% For example stims, show individual fly responses
# cv := std across animals normalized by mean across animals and across all stims for that glom
# cv = np.std(response_amplitudes, axis=-1) / np.mean(response_amplitudes, axis=(1, 2))[:, None]

# Normalize sigma by the maximum response of this glom across all stims
scaling = np.nanmax(np.nanmean(response_amplitudes, axis=-1), axis=-1)
cv = np.nanstd(response_amplitudes, axis=-1) / scaling[:, None]
eg_leaf_inds = [7, 9, 12]  # [7, 9, 12]
eg_stim_ind = 6  # 6
fh2, ax2 = plt.subplots(len(eg_leaf_inds), all_responses.shape[-1], figsize=(3.5, 2.5))

[x.set_ylim([-0.1, 0.5]) for x in ax2.ravel()]
[x.set_axis_off() for x in ax2.ravel()]
for li, leaf_ind in enumerate(eg_leaf_inds):
    g_ind = leaves[leaf_ind]
    for fly_ind in range(all_responses.shape[-1]):

        ax2[li, fly_ind].plot(response_data.get('time_vector'), all_responses[g_ind, :, eg_stim_ind, fly_ind],
                              color='k', alpha=0.5)
        ax2[li, fly_ind].plot(response_data.get('time_vector'), np.mean(all_responses[g_ind, :, eg_stim_ind, :], axis=-1),
                              color=colors[g_ind, :], linewidth=1, alpha=1.0)
        if fly_ind == 0 & li == 0:
            plot_tools.addScaleBars(ax2[0, 0], dT=2, dF=0.25, T_value=0, F_value=-0.05)

        if fly_ind == 0:
            ax2[li, fly_ind].annotate('cv = {:.2f}'.format(cv[g_ind, eg_stim_ind]), (0, 0.5))

print(np.array(included_gloms)[np.array([leaves[x] for x in eg_leaf_inds])])
fh2.savefig(os.path.join(save_directory, 'pgs_fly_responses.svg'), transparent=True)

fh3, ax3 = plt.subplots(1, 1, figsize=(3.5, 2.5))
ax3.hist(cv.ravel(), bins=40)
ax3.set_xlabel('Coefficient of Variation')
ax3.set_ylabel('Count');

fh2.savefig(os.path.join(save_directory, 'pgs_fly_responses.svg'), transparent=True)

fh4, ax4 = plt.subplots(1, 1, figsize=(3.5, 2.5))
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    mean_cv = np.mean(cv[g_ind, :])
    std_cv = np.std(cv[g_ind, :])
    sem_cv = np.std(cv[g_ind, :]) / np.sqrt(len(cv[g_ind, :]))

    ax4.plot(leaf_ind * np.ones_like(cv[g_ind, :]), cv[g_ind, :], color='k', marker='.', linestyle='none', alpha=0.25, markersize=4)
    ax4.plot(leaf_ind, mean_cv, marker='o', color=colors[g_ind, :])
    ax4.plot([leaf_ind, leaf_ind], [mean_cv-sem_cv, mean_cv+sem_cv], marker='None', color=colors[g_ind, :])

ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels([included_gloms[x] for x in leaves])
ax4.set_ylabel('Coefficient of variation', fontsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11, rotation=90)
# ax4.set_ylim([0, np.nanmax(cv.ravel())])
# ax4.set_ylim([0, 1])

fh4.savefig(os.path.join(save_directory, 'pgs_fly_cv.svg'), transparent=True)

# %% Trial-to-trial variability


fh5, ax5 = plt.subplots(len(eg_leaf_inds), 5, figsize=(3.0, 2))

[x.set_ylim([-0.2, 1.2]) for x in ax5.ravel()]
[x.set_axis_off() for x in ax5.ravel()]
for li, leaf_ind in enumerate(eg_leaf_inds):
    g_ind = leaves[leaf_ind]

    for trial in range(5):
        ax5[li, trial].plot(trial_response_by_stimulus[eg_stim_ind][g_ind, :, trial], color='k', alpha=0.5)
        ax5[li, trial].plot(np.mean(trial_response_by_stimulus[eg_stim_ind][g_ind, :, :], axis=-1), color=colors[g_ind, :])

        if trial == 0 & li == 0:
            plot_tools.addScaleBars(ax5[0, 0], dT=2, dF=0.25, T_value=0, F_value=-0.08)

fh5.savefig(os.path.join(save_directory, 'pgs_trial_responses.svg'), transparent=True)



# %% Inter-individual correlation for each glom

fh4, ax4 = plt.subplots(1, 1, figsize=(3.0, 2))
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    inter_corr = pd.DataFrame(response_amplitudes[g_ind, :, :]).corr().to_numpy()[np.triu_indices(response_amplitudes.shape[-1], k=1)]
    mean_inter_corr = np.mean(inter_corr)
    std_inter_corr = np.std(inter_corr)
    sem_inter_corr = np.std(inter_corr) / np.sqrt(len(inter_corr))
    ax4.plot(leaf_ind * np.ones_like(inter_corr), inter_corr, color=colors[g_ind, :], marker='.', linestyle='none', alpha=0.5, markersize=4)
    ax4.plot(leaf_ind, mean_inter_corr, color=colors[g_ind, :], marker='o', markersize=6)
    ax4.plot([leaf_ind, leaf_ind], [mean_inter_corr-sem_inter_corr, mean_inter_corr+sem_inter_corr], marker='None', color=colors[g_ind, :])

ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels([included_gloms[x] for x in leaves])
ax4.set_ylabel('Inter-individual corr. (r)', fontsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11, rotation=90)
# ax4.set_ylim([0, 1])

fh4.savefig(os.path.join(save_directory, 'pgs_Inter_Ind_Corr.svg'), transparent=True)


# %% OTHER STUFF

# %% Visualize clustering of individual gloms, with aligned glom identiy overlaid

# X: individual gloms x features
# Row order = LCa_fly1, LCa_fly2, ..., LCb_fly1, LCb_fly2, ... LCn_flyn
glom_ids = np.repeat(included_vals, response_amplitudes.shape[-1], axis=0)
glom_names = np.array([np.array(included_gloms)[x == included_vals][0] for x in glom_ids])
# X: response traces
all_concat = np.hstack([all_responses[:, :, x, :] for x in range(30)])
X = np.reshape(np.swapaxes(all_concat, 1, -1), (-1, all_concat.shape[1]))

# Remove individual gloms that were not included in that fly (vals replaced by nan)
nan_glom_ind = np.any(np.isnan(X), axis=1)
glom_ids = np.delete(glom_ids, nan_glom_ind, axis=0)
glom_names = np.delete(glom_names, nan_glom_ind, axis=0)
X = np.delete(X, nan_glom_ind, axis=0)


row_colors = cmap(glom_ids/glom_ids.max())
X = zscore(X, axis=1)

# Compute linkage matrix, Z
Z = linkage(X,
            method='average',
            metric='euclidean',
            optimal_ordering=True)

# DENDROGRAM
fh5, ax5 = plt.subplots(1, 2, figsize=(4, 9), gridspec_kw={'width_ratios': [1, 3], 'wspace': 0.01})
[util.cleanAxes(x) for x in ax5.ravel()]

with plt.rc_context({'lines.linewidth': 0.25}):
    D = dendrogram(Z, p=len(glom_ids), truncate_mode='lastp', ax=ax5[0],
                   above_threshold_color='black',
                   color_threshold=1,
                   orientation='left',
                   leaf_rotation=0,
                   leaf_font_size=10,
                   count_sort=True,
                   labels=glom_ids)

ind_glom_leaves = glom_ids[D.get('leaves')]
ind_glom_leaf_colors = row_colors[D.get('leaves')]
ind_glom_leaves
included_gloms

ylbls = ax5[0].get_ymajorticklabels()
for l_ind, lbl in enumerate(ylbls):
    lbl.set_color(cmap(int(lbl.get_text())/glom_ids.max()))
# ax5[0].set_yticklabels([r'$\blacksquare$' for x in range(len(glom_names))], fontweight='bold', fontsize=3)
# ax5[0].set_yticklabels([r'$\blacksquare$ ' + glom_names[D.get('leaves')][x] for x in range(len(glom_names))], fontweight='bold', fontsize=10)

dx = 1
dy = 1

# Plot all color rects
for leaf_ind, leaf in enumerate(ind_glom_leaves):
    x0 = 0
    y0 = leaf_ind
    color = ind_glom_leaf_colors[leaf_ind, :]
    rect = Rectangle((x0, y0), dx, dy, color=color)
    ax5[1].add_patch(rect)

for glom_ind, glom_val in enumerate(included_vals):
    for leaf_ind, leaf in enumerate(ind_glom_leaves):
        x0 = 1+glom_ind
        y0 = leaf_ind

        if leaf == glom_val:
            color = ind_glom_leaf_colors[leaf_ind, :]
        else:
            color = [0.6, 0.6, 0.6, 1.0]
        rect = Rectangle((x0, y0), dx, dy, color=color)
        ax5[1].add_patch(rect)

disp_gloms = included_gloms.copy()
disp_gloms.insert(0, 'All')

ax5[1].set_xlim([0, glom_ind+dx+1])
ax5[1].set_ylim([0, leaf_ind+dy])
ax5[1].set_xticks(np.arange(0, len(disp_gloms))+0.5)
ax5[1].set_xticklabels(disp_gloms, rotation=90)
ax5[1].set_yticks([])
ax5[1].xaxis.tick_top()

ax5[0].get_yaxis().set_visible(False)

fh5.savefig(os.path.join(save_directory, 'pgs_ind_dendrogram.svg'), transparent=True)
