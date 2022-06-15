from visanalysis.analysis import imaging_data, shared_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from visanalysis.util import plot_tools
from sklearn.metrics import explained_variance_score
from scipy.optimize import least_squares
from scipy.stats import pearsonr, zscore, multivariate_normal, spearmanr
from sklearn.decomposition import PCA


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

eg_series = ('2022-03-01', 2)

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))

included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# %%
all_cmats = []
all_cmats_shuffled = []
r_gain_behavior = []
all_gain_by_trial = []
has_beh_inds = []
all_beh = []
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
    gain_by_trial = np.zeros((len(included_gloms), epoch_response_matrix.shape[1]))  # gloms x trials
    gain_by_trial[:] = np.nan

    gain_by_stim = np.zeros((len(included_gloms),
                              len(unique_parameter_values),
                              len(pull_inds[0])))  # gloms x stim condition x trials
    gain_by_stim[:] = np.nan

    cmats = []
    cmats_shuffled = []
    for up, pind in enumerate(pull_inds):
        gain_for_stim = response_amplitude[:, pind] / np.nanmean(response_amplitude[:, pind], axis=1)[:, np.newaxis]
        gain_by_stim[:, up, :] = gain_for_stim
        gain_by_trial[:, pind] = gain_for_stim
        new_cmat = np.corrcoef(gain_for_stim)  # glom x glom. Corr map for this stim type
        cmats.append(new_cmat)

        # Trial shuffle to remove corrs
        iterations = 100
        new_cmat_shuffled = []
        for it in range(iterations):
            idx = np.random.rand(*gain_for_stim.T.shape).argsort(0)
            gain_for_stim_shuffled = gain_for_stim.T[idx, np.arange(gain_for_stim.T.shape[1])].T

            new_cmat_shuffled.append(np.corrcoef(gain_for_stim_shuffled))
        new_cmat_shuffled = np.mean(np.stack(new_cmat_shuffled, axis=-1), axis=-1)  # average across iterations
        cmats_shuffled.append(new_cmat_shuffled)

    all_cmats.append(np.stack(cmats, axis=-1))  # stack across stims
    all_cmats_shuffled.append(np.stack(cmats_shuffled, axis=-1))

    # Load behavior data
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    if ft_data_path:
        has_beh_inds.append(s_ind)
        print('HAS BEHAVIOR')
        behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                                 response_len = response_data.get('response').shape[1],
                                                 process_behavior=True, fps=50, exclude_thresh=300)

        r_gain_behavior_new = np.zeros((len(included_gloms), len(pull_inds)))
        r_gain_behavior_new[:] = np.nan
        all_beh.append(behavior_data['walking_amp'][0, :])
        for up, pind in enumerate(pull_inds):
            beh = behavior_data['walking_amp'][0, pind]  # for all trials of this stim type
            for g_ind, glom in enumerate(included_gloms):
                gn = gain_by_stim[g_ind, up, :]
                r = np.corrcoef(beh, gn)[0, 1]
                r_gain_behavior_new[g_ind, up] = r

        r_gain_behavior.append(r_gain_behavior_new)

    all_gain_by_trial.append(gain_by_trial)
    print('--------')
all_cmats = np.stack(all_cmats, axis=-1)  # (glom x glom x stim x fly)
all_cmats_shuffled = np.stack(all_cmats_shuffled, axis=-1)
r_gain_behavior = np.stack(r_gain_behavior, axis=-1)  # (gloms x stim x fly)
all_gain_by_trial = np.stack(all_gain_by_trial, axis=-1)  # (gloms x trial x fly)
all_beh = np.stack(all_beh, axis=-1)  # (time x fly) for subset of flies with behavior


# %% # Plot eg trials for eg fly
eg_fly_ind = 4
eg_trials = np.arange(12, 20)
eg_glom_inds = [0, 3, 11]

series = matching_series[eg_fly_ind]
series_number = series['series']
file_path = series['file_name'] + '.hdf5'
file_name = os.path.split(series['file_name'])[-1]
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# Load response data
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

# Align responses
unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix)

# Calculate gain for each trial. Gain := response amplitude normalized by median response amplitude to that stim
parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
pull_inds = [np.where([pv == up for pv in parameter_values])[0] for up in unique_parameter_values]
response_amplitude = ID.getResponseAmplitude(epoch_response_matrix, metric='max')  # gloms x trials

stim_inds = [1, 2, 3, 4, 5]  # exclude d gratings, not much responses
fh0, ax0 = plt.subplots(len(eg_glom_inds), len(stim_inds), figsize=(5.5, 2.5), gridspec_kw={'wspace': 0.025, 'hspace':0.025})

for up_ind, up in enumerate(stim_inds):
    concat_trial_response = np.concatenate([trial_response_by_stimulus[up][:, x, :] for x in eg_trials], axis=1)
    concat_time = np.arange(concat_trial_response.shape[1]) * ID.getAcquisitionMetadata('sample_period')
    [x.set_ylim([-0.15, 1.1]) for x in ax0.ravel()]
    [util.clean_axes(x) for x in ax0.ravel()]
    [x.set_ylim() for x in ax0.ravel()]
    for idx, eg_glom_ind in enumerate(eg_glom_inds):
        glom = included_gloms[eg_glom_ind]
        if up_ind == 0:
            ax0[idx, up_ind].set_ylabel(glom, rotation=90)
            if idx == 0:
                plot_tools.addScaleBars(ax0[idx, up_ind], dT=4, dF=0.25, T_value=-0.5, F_value=-0.1)
        if idx == 0:
            trial_len = trial_response_by_stimulus[0].shape[2]
            concat_len = concat_trial_response.shape[1]
            y_val = 1.1
            ax0[idx, up_ind].plot(np.linspace(trial_len/2,
                                                concat_len-trial_len/2,
                                                len(eg_trials))* ID.getAcquisitionMetadata('sample_period'),
                                    y_val * np.ones(len(eg_trials)),
                                    'rv')
        ax0[idx, up_ind].plot(concat_time, concat_trial_response[eg_glom_ind, :], color=util.get_color_dict()[glom])

fh0.savefig(os.path.join(save_directory, 'pgs_variance_eg_fly.svg'), transparent=True)

# %%
# Plot response clouds pairs of gloms

def getGaussianPDF(x_vals, y_vals, N=100):
    X = np.linspace(x_vals.min(), x_vals.max(), N)
    Y = np.linspace(y_vals.min(), y_vals.max(), N)
    X, Y = np.meshgrid(X, Y)

    mu = np.array([np.mean(x_vals), np.mean(y_vals)])
    Sigma = np.cov(x_vals, y_vals)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    Z = Z / Z.max()

    return X, Y, Z


stim_ind = 2
tmp_amps = ID.getResponseAmplitude(trial_response_by_stimulus[stim_ind], metric='max')

fh1, ax1 = plt.subplots(len(eg_glom_inds), len(eg_glom_inds), figsize=(4.5, 4), tight_layout=True)
[x.spines['top'].set_visible(False) for x in ax1.ravel()]
[x.spines['right'].set_visible(False) for x in ax1.ravel()]
[x.set_xlim([-0.05, 1.2]) for x in ax1.ravel()]
[x.set_ylim([-0.05, 1.2]) for x in ax1.ravel()]
[x.set_axis_off() for x in ax1.ravel()]
for idx_1, glom_1_ind in enumerate(eg_glom_inds):
    for idx_2, glom_2_ind in enumerate(eg_glom_inds):
        if idx_1 > idx_2:
            ax1[idx_1, idx_2].set_axis_on()
            if idx_1 == 2:
                ax1[idx_1, idx_2].set_xlabel(included_gloms[glom_2_ind])
            if idx_2 == 0:
                ax1[idx_1, idx_2].set_ylabel(included_gloms[glom_1_ind])
            ax1[idx_1, idx_2].plot(tmp_amps[glom_1_ind, :], tmp_amps[glom_2_ind, :],
                                   color='k', alpha=0.5, marker='.', linestyle='None')

            X, Y, Z = getGaussianPDF(tmp_amps[glom_1_ind, :], tmp_amps[glom_2_ind, :], N=100)
            ax1[idx_1, idx_2].contour(X, Y, Z, colors='k', levels=1, linewidths=1)
            r = np.corrcoef(tmp_amps[glom_1_ind, :], tmp_amps[glom_2_ind, :])[0, 1]
            ax1[idx_1, idx_2].annotate('r = {:.2f}'.format(r), (0.7, 0.05))
            ax1[idx_1, idx_2].set_xticks([0, 0.5, 1.0])
            ax1[idx_1, idx_2].set_yticks([0, 0.5, 1.0])
            ax1[idx_1, idx_2].set_xticklabels([0, '', 1.0])
            ax1[idx_1, idx_2].set_yticklabels([0, '', 1.0])


# ax1[2, 0].set_xticklabels([0, '', 1.0])
# ax1[2, 0].set_yticklabels([0, '', 1.0])

fh1.supylabel('Glom 1 trial response (dF/F)')
fh1.supxlabel('Glom 2 trial response (dF/F)')

fh1.savefig(os.path.join(save_directory, 'pgs_variance_eg_responseclouds.svg'), transparent=True)

# %% glom-glom correlation matrix, across all stims
fly_average_cmats = np.nanmean(all_cmats, axis=-1)  # shape = glom x glom x stim
fly_average_cmats_shuffle = np.nanmean(all_cmats_shuffled, axis=-1)  # shape = glom x glom x stim

glom_corr_df = pd.DataFrame(data=np.nanmean(fly_average_cmats, axis=-1), index=included_gloms, columns=included_gloms)
glom_corr_df_shuffle = pd.DataFrame(data=np.nanmean(fly_average_cmats_shuffle, axis=-1), index=included_gloms, columns=included_gloms)
fh1, ax1 = plt.subplots(1, 1, figsize=(2.75, 2.25))
g=sns.heatmap(glom_corr_df,
            ax=ax1,
            vmin=0, vmax=+1,
            cmap='Reds',
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Gain-gain corr. (r)'},
            rasterized=True
            )
ax1.tick_params(axis='both', which='major', labelsize=8)

fh1.savefig(os.path.join(save_directory, 'pgs_variance_corrmat.svg'), transparent=True)

fh2, ax2 = plt.subplots(2, 1, figsize=(0.9, 2), tight_layout=True)
sns.heatmap(glom_corr_df,
            ax=ax2[0],
            vmin=0, vmax=+1,
            cmap='Reds',
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cbar_kws={'label': 'Gain-gain corr. (r)'},
            rasterized=False
            )
ax2[0].set_title('Unshuffled')
sns.heatmap(glom_corr_df_shuffle,
            ax=ax2[1],
            vmin=0, vmax=+1,
            cmap='Reds',
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cbar_kws={'label': 'Gain-gain corr. (r)'},
            rasterized=False
            )
ax2[1].set_title('Trial shuffled')
fh2.savefig(os.path.join(save_directory, 'pgs_variance_cmatvsshuffle.svg'), transparent=True)

# Corr mat for each stim type
fh3, ax3 = plt.subplots(2, 3, figsize=(7, 4.5))
ax3 = ax3.ravel()
cbar_ax = fh3.add_axes([.91, .3, .03, .4])
for s in range(fly_average_cmats.shape[-1]):
    if s == 3:
        xticklabels = included_gloms
        yticklabels = included_gloms
    else:
        xticklabels = False
        yticklabels = False

    if s == 5:
        cbar = True
    else:
        cbar = False


    g=sns.heatmap(fly_average_cmats[..., s],
                ax=ax3[s],
                vmin=0, vmax=+1,
                cmap='Reds',
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=cbar,
                cbar_ax=cbar_ax,
                cbar_kws={'label': 'Gain-gain corr. (r)'},
                rasterized=True
                )
fh3.savefig(os.path.join(save_directory, 'pgs_variance_cmat_per_stim.svg'), transparent=True)


# %% For flies w behavior tracking: corr between gain & behavior, for each stim and glom

df = pd.DataFrame(data=np.nanmean(r_gain_behavior, axis=-1),
                  index=included_gloms, columns=[str(s) for s in unique_parameter_values])

fh2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
sns.heatmap(df,
            cmap='coolwarm', vmin=-0.82, vmax=+0.82,
            ax=ax2,
            yticklabels=True,
            xticklabels=False,
            cbar_kws={'label': 'Behavior-gain corr (r)'})

fh2.savefig(os.path.join(save_directory, 'pgs_variance_behcorr.svg'), transparent=True)



# %% Covariance analysis

fh4, ax4 = plt.subplots(len(has_beh_inds), 1, figsize=(12, 6))

first_pcs = []
frac_var_explained = []
z_beh = []
z_proj = []
for b_ind, f_ind in enumerate(has_beh_inds):
    new_beh = all_beh[:, b_ind]

    gain_by_trial = np.nan_to_num(all_gain_by_trial[..., f_ind])
    pca = PCA(n_components=13).fit(gain_by_trial.T)
    frac_var_explained.append(pca.explained_variance_ratio_)
    pca = PCA(n_components=1).fit(gain_by_trial.T)
    pc = pca.components_[0, :]
    first_pcs.append(pc)

    # some LC17 are excluded. Don't use this glom for projection...
    nan_rows = np.any(np.isnan(gain_by_trial), axis=1)
    proj = np.dot(pc[~nan_rows], gain_by_trial[~nan_rows, :])
    ax4[b_ind].plot(new_beh, 'k')
    twax = ax4[b_ind].twinx()
    twax.plot(proj, 'b')

    z_beh.append(zscore(new_beh))
    z_proj.append(zscore(proj))

frac_var_explained = np.stack(frac_var_explained, axis=-1)
first_pcs = np.stack(first_pcs, axis=-1)
z_beh = np.stack(z_beh, axis=-1)
z_proj = np.stack(z_proj, axis=-1)

fh3, ax3 = plt.subplots(1, 1, figsize=(2, 2))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.errorbar(x=np.arange(1, 1+len(included_gloms)),
             y=np.mean(frac_var_explained, axis=-1),
             yerr=np.std(frac_var_explained, axis=-1) / np.sqrt(frac_var_explained.shape[-1]),
             color='k', marker='o')
ax3.set_ylabel('Frac. var. explained')
ax3.set_xlabel('PCs')

fh3.savefig(os.path.join(save_directory, 'pgs_variance_pca_fracvar.svg'), transparent=True)

 #%%
fh4, ax4 = plt.subplots(1, 1, figsize=(3.25, 2.0))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
mean_pc1 = np.mean(first_pcs, axis=-1)
err_pc1 = np.std(first_pcs, axis=-1) / np.sqrt(first_pcs.shape[-1])
ax4.bar(x=included_gloms,
        height=mean_pc1,
        color='k')
plt.xticks(rotation=90)
ax4.errorbar(x=included_gloms,
            y=mean_pc1,
            yerr=err_pc1,
            color='k',
            linestyle='None')


ax4.set_title('First PC')
fh4.savefig(os.path.join(save_directory, 'pgs_variance_pca_pc1.svg'), transparent=True)


# %%

fh5, ax5 = plt.subplots(1, len(has_beh_inds), figsize=(7, 1))

[x.spines['top'].set_visible(False) for x in ax5.ravel()]
[x.spines['right'].set_visible(False) for x in ax5.ravel()]

[x.set_xlim([1.1*np.nanmin(z_beh), 1.1*np.nanmax(z_beh)]) for x in ax5.ravel()]
[x.set_ylim([1.1*np.nanmin(z_proj), 1.1*np.nanmax(z_proj)]) for x in ax5.ravel()]
corr_vals = []
for b_ind in range(len(has_beh_inds)):
    ax5[b_ind].axhline(y=0, color='k', alpha=0.5)
    ax5[b_ind].axvline(x=0, color='k', alpha=0.5)
    ax5[b_ind].plot(z_beh[:, b_ind], z_proj[:, b_ind], 'k.')
    r, p = spearmanr(z_beh[:, b_ind], z_proj[:, b_ind])
    corr_vals.append(r)
    ax5[b_ind].set_title(r'$\rho$ = {:.2f}'.format(r))

[x.set_xticks([-2, 0, 2, 4]) for x in ax5]
[x.set_yticks([-2, 0, 2, 4]) for x in ax5]
[x.set_xticklabels([]) for x in ax5[1:]]
[x.set_yticklabels([]) for x in ax5[1:]]

fh5.supxlabel('Behavior amplitude (z-score)')
fh5.supylabel('First principal \ncomponent proj. \n(z-score)', ha='center')

fh5.savefig(os.path.join(save_directory, 'pgs_variance_pc_beh_corr.svg'), transparent=True)

# %%
plt.plot(z_beh.ravel(), z_proj.ravel(), 'k.', alpha=0.25)
r, p = spearmanr(z_beh.ravel(), z_proj.ravel())



# %%

fh4, ax4 = plt.subplots(1, 1, figsize=(3, 2))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.bar(np.arange(len(included_gloms)), np.nanmean(first_pcs, axis=1),
        color='k')
ax4.set_xticks(np.arange(len(included_gloms)))
ax4.set_xticklabels(included_gloms, rotation=90)
ax4.set_ylabel('Variance along \n glom response axis')
ax4.set_title('First PC')

fh4.savefig(os.path.join(save_directory, 'pgs_variance_pca_glom_pc1.svg'), transparent=True)



# %%
