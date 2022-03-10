from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd
import ants
import seaborn as sns

from scipy.stats import zscore

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, set_link_color_palette, fcluster


from glom_pop import dataio, util, alignment

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']

# TODO: Tuning to some params:
#       -light/dark
#       -L/R movement
#       -Spot size

# %% MEAN RESPONSES TO TUNING SUITE

glom_size_threshold = 10

# Load overall glom map
vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
glom_mask_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')).numpy()
all_vals, all_names = dataio.get_glom_mask_decoder(glom_mask_2_meanbrain)

all_sizes = pd.DataFrame(data=[np.sum(glom_mask_2_meanbrain == mv) for mv in all_vals],
                         index=all_names.values)

# Get included gloms
included_gloms = dataio.get_included_gloms()
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# Get all glom mask values & names
vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
all_glom_values = vpn_types['Unnamed: 0'].values
all_glom_names = vpn_types['vpn_types'].values

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                                          'include_in_analysis': True})

# %%
all_responses = []
response_amplitudes = []
all_glom_sizes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    # epoch_response_matrix: shape=(included_gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    glom_sizes = np.zeros(len(all_glom_values))  # Glom sizes of ALL gloms, not just included
    for val_ind, new_val in enumerate(all_glom_values):
        new_glom_size = np.sum(response_data.get('mask') == new_val)
        glom_sizes[val_ind] = new_glom_size

        if new_val in included_vals:  # Append responses for included gloms only
            included_ind = np.where(new_val == included_vals)[0]
            if new_glom_size > glom_size_threshold:
                pull_ind = np.where(new_val == response_data.get('mask_vals'))[0][0]
                epoch_response_matrix[included_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
            else:  # Exclude because this glom, in this fly, is too tiny
                pass

    # Align responses
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

    response_amp = ID.getResponseAmplitude(mean_response, metric='max')

    all_responses.append(mean_response)
    response_amplitudes.append(response_amp)
    all_glom_sizes.append(glom_sizes)
    del response_amp, mean_response, glom_sizes


# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)
all_glom_sizes = np.stack(all_glom_sizes, axis=-1)  # dims = (gloms, fly)

# Exclude last two stims (full field flashes)
unique_parameter_values = unique_parameter_values[:-2]
all_responses = all_responses[:, :-2, :, :]
response_amplitudes = response_amplitudes[:, :-2, :]

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

# Concatenate responses across stimulus types
#   Shape = (glom, concat time)
mean_concat = np.vstack(np.concatenate([mean_responses[:, x, :] for x in np.arange(len(unique_parameter_values))], axis=1))
sem_concat = np.vstack(np.concatenate([sem_responses[:, x, :] for x in np.arange(len(unique_parameter_values))], axis=1))
std_concat = np.vstack(np.concatenate([std_responses[:, x, :] for x in np.arange(len(unique_parameter_values))], axis=1))
time_concat = np.arange(0, mean_concat.shape[-1]) * ID.getAcquisitionMetadata().get('sample_period')

np.save(os.path.join(save_directory, 'chat_responses.npy'), all_responses)
np.save(os.path.join(save_directory, 'chat_response_amplitudes.npy'), response_amplitudes)
np.save(os.path.join(save_directory, 'mean_chat_responses.npy'), mean_responses)
np.save(os.path.join(save_directory, 'sem_chat_responses.npy'), sem_responses)
np.save(os.path.join(save_directory, 'included_gloms.npy'), included_gloms)

# %% QC: Number of voxels in each glomerulus. Included vs. excluded gloms sizes

glom_sizes_pd = pd.DataFrame(data=all_glom_sizes.copy(),
                             index=all_glom_names)
# Drop gloms with all zeros
glom_sizes_pd = glom_sizes_pd.loc[~(glom_sizes_pd == 0).all(axis=1)]
nonzero_glom_names = glom_sizes_pd.index.values
nonzero_glom_vals = dataio.get_glom_vals_from_names(nonzero_glom_names)

fh, ax = plt.subplots(1, 1, figsize=(5, 3))
for ind, ig in enumerate(glom_sizes_pd.index):
    if ig in included_gloms:
        color = 'k'
    else:
        color = 'r'
    ax.plot(ind*np.ones(glom_sizes_pd.shape[1]),
            glom_sizes_pd.loc[ig, :],
            color=color, marker='.', linestyle='none')

ax.set_ylabel('Number of voxels')
ax.axhline(glom_size_threshold, color='k', linestyle='-', alpha=0.5)
ax.set_xticks(np.arange(0, len(glom_sizes_pd.index)))
ax.set_xticklabels(glom_sizes_pd.index, rotation=90);
ax.set_yscale('log')
fh.savefig(os.path.join(save_directory, 'pgs_glom_sizes.svg'), transparent=True)

# %% QC: glom response traces. For MC or occlusion artifacts

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'

    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    fh, ax = plt.subplots(len(nonzero_glom_vals), 1, figsize=(8, 12))
    ax[0].set_title('{}: {}'.format(os.path.split(file_path)[-1], series_number))

    for g_ind, g_val in enumerate(nonzero_glom_vals):
        pull_ind = np.where(g_val == response_data['mask_vals'])[0][0]

        ax[g_ind].plot(response_data['response'][pull_ind, :])
        ax[g_ind].set_ylabel(dataio.get_glom_name_from_val(g_val))

# %%

# # # Cluster on stimulus tuning # # #
# Average response amplitudes across flies...
peak_responses = np.nanmean(response_amplitudes, axis=-1)  # (glom x stim)
peak_responses = zscore(peak_responses, axis=1)

# Compute linkage matrix, Z
Z = linkage(peak_responses,
            method='complete',
            metric='euclidean',
            optimal_ordering=True)

clusters = fcluster(Z, t=5, criterion='distance')

# Colors := desaturated primaries, one for each cluster
colors = ['b',
          'g',
          'y',
          'm']

set_link_color_palette(colors)
# DENDROGRAM
fh0, ax0 = plt.subplots(1, 1, figsize=(1.5, 6))
D = dendrogram(Z, p=len(included_gloms), truncate_mode='lastp', ax=ax0,
               above_threshold_color='black',
               color_threshold=5.0, orientation='left',
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
[util.clean_axes(x) for x in ax1.ravel()]

fh1.subplots_adjust(wspace=0.00, hspace=0.00)
# for u_ind, un in enumerate(unique_parameter_values):
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    # ax1[leaf_ind].set_ylabel(name, fontsize=11, rotation=0)
    # ax1[leaf_ind].fill_between(time_concat,
    #                            mean_concat[g_ind, :]-sem_concat[g_ind, :],
    #                            mean_concat[g_ind, :]+sem_concat[g_ind, :],
    #                            facecolor=sns.desaturate(colors[g_ind, :], 0.25), alpha=1.0, linewidth=0,
    #                            rasterized=True)
    ax1[leaf_ind].plot(time_concat, mean_concat[g_ind, :],
                       color=util.get_color_dict().get(name), alpha=1.0, linewidth=1.0,
                       rasterized=True)

    if (leaf_ind == 0):
        plot_tools.addScaleBars(ax1[leaf_ind], dT=2, dF=0.25, T_value=0, F_value=-0.1)

[x.set_ylim([mean_responses.min(), 1.1*mean_responses.max()]) for x in ax1.ravel()]

fh0.savefig(os.path.join(save_directory, 'pgs_tuning_dendrogram.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'pgs_mean_tuning.svg'), transparent=True, dpi=300)
# plt.get_cmap('tab20b')
# Save leaves list and ordered response dataframe
np.save(os.path.join(save_directory, 'cluster_leaves_list.npy'), leaves)
np.save(os.path.join(save_directory, 'cluster_vals.npy'), clusters)
concat_df = pd.DataFrame(data=mean_concat[leaves, :], index=np.array(included_gloms)[leaves])
concat_df.to_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl'))


# %% glom highlight maps

# Load mask key for VPN types

glom_mask_2_meanbrain = alignment.filter_glom_mask_by_name(mask=glom_mask_2_meanbrain,
                                                           vpn_types=vpn_types,
                                                           included_gloms=included_gloms)

fh6, ax6 = plt.subplots(len(np.unique(clusters)), 1, figsize=(1.5, 9))
[x.set_xlim([30, 230]) for x in ax6.ravel()]
[x.set_ylim([180, 5]) for x in ax6.ravel()]
[x.set_axis_off() for x in ax6.ravel()]

for c in np.unique(clusters):
    highlight_names = list(np.array(included_gloms)[clusters==c])
    util.make_glom_map(ax=ax6[c-1],
                       glom_map=glom_mask_2_meanbrain,
                       z_val=None,
                       highlight_names=highlight_names)

# for leaf_ind, g_ind in enumerate(leaves):
#     name = included_gloms[g_ind]
#     # glom_mask_val = vpn_types.loc[vpn_types.get('vpn_types') == name, 'Unnamed: 0'].values[0]
#
#     util.make_glom_map(ax=ax6[leaf_ind],
#                        glom_map=glom_mask_2_meanbrain,
#                        z_val=None,
#                        highlight_names=[name])

# fh6.savefig(os.path.join(save_directory, 'pgs_glom_highlights.svg'), transparent=True)

# %% fly-fly variability for each stim

# re-order by leaves
cv = pd.DataFrame(data=np.nanstd(response_amplitudes, -1) / np.nanmean(response_amplitudes, -1),
                  index=included_gloms).iloc[leaves, :]

fh7, ax7 = plt.subplots(1, 1, figsize=(4, 2.5))
sns.heatmap(cv, vmin=0, cmap='Greys',
            xticklabels=False, yticklabels=True,
            cbar_kws={'label': 'Coefficient of variation'},
            ax=ax7)

ax7.set_title('Fly to fly variability')

fh7.savefig(os.path.join(save_directory, 'pgs_fly_cv.svg'), transparent=True)



# %% For example stims, show individual fly responses
# cv := std across animals normalized by mean across animals and across all stims for that glom

# Normalize sigma by the maximum response of this glom across all stims
scaling = np.nanmax(np.nanmean(response_amplitudes, axis=-1), axis=-1)
# cv = np.nanstd(response_amplitudes, axis=-1) / scaling[:, None]
cv = np.nanstd(response_amplitudes, -1) / np.nanmean(response_amplitudes, -1)
eg_leaf_inds = [2, 6, 8]
eg_stim_ind = 8  # 8
fh2, ax2 = plt.subplots(len(eg_leaf_inds), all_responses.shape[-1], figsize=(4.0, 2.5))
print(unique_parameter_values[eg_stim_ind])
[x.set_ylim([-0.1, 0.65]) for x in ax2.ravel()]
[x.set_axis_off() for x in ax2.ravel()]
for li, leaf_ind in enumerate(eg_leaf_inds):
    g_ind = leaves[leaf_ind]
    for fly_ind in range(all_responses.shape[-1]):
        ax2[li, fly_ind].plot(response_data.get('time_vector'), all_responses[g_ind, eg_stim_ind, :, fly_ind],
                              color='k', alpha=0.5)
        ax2[li, fly_ind].plot(response_data.get('time_vector'), np.mean(all_responses[g_ind, eg_stim_ind, :, :], axis=-1),
                              color=util.get_color_dict()[included_gloms[g_ind]], linewidth=1, alpha=1.0)
        if fly_ind == 0 & li == 0:
            plot_tools.addScaleBars(ax2[0, 0], dT=2, dF=0.25, T_value=0, F_value=-0.05)

        if fly_ind == 0:
            ax2[li, fly_ind].annotate('{}'.format(np.array(included_gloms)[g_ind]), (0, 0.35))

fh2.savefig(os.path.join(save_directory, 'pgs_fly_responses.svg'), transparent=True)




# %% Inter-individual correlation for each glom
# TODO: check this out...
fh4, ax4 = plt.subplots(1, 1, figsize=(3.0, 2))
for leaf_ind, g_ind in enumerate(leaves):
    name = included_gloms[g_ind]
    inter_corr = pd.DataFrame(response_amplitudes[g_ind, :, :]).corr().to_numpy()[np.triu_indices(response_amplitudes.shape[-1], k=1)]
    mean_inter_corr = np.nanmean(inter_corr)
    std_inter_corr = np.std(inter_corr)
    sem_inter_corr = np.std(inter_corr) / np.sqrt(len(inter_corr))
    ax4.plot(leaf_ind * np.ones_like(inter_corr), inter_corr, color=util.get_color_dict()[included_gloms[g_ind]], marker='.', linestyle='none', alpha=0.5, markersize=4)
    ax4.plot(leaf_ind, mean_inter_corr, color=util.get_color_dict()[included_gloms[g_ind]], marker='o', markersize=6)
    ax4.plot([leaf_ind, leaf_ind], [mean_inter_corr-sem_inter_corr, mean_inter_corr+sem_inter_corr], marker='None', color=util.get_color_dict()[included_gloms[g_ind]])

ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels([included_gloms[x] for x in leaves])
ax4.set_ylabel('Inter-individual corr. (r)', fontsize=11)
ax4.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11, rotation=90)
# ax4.set_ylim([0, 1])
inter_corr
fh4.savefig(os.path.join(save_directory, 'pgs_Inter_Ind_Corr.svg'), transparent=True)


# %% OTHER STUFF

# %% Visualize clustering of individual gloms, with aligned glom identity overlaid

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
[util.clean_axes(x) for x in ax5.ravel()]

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
