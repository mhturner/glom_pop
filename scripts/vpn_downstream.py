from neuprint import (Client, fetch_adjacencies, NeuronCriteria, fetch_neurons,
                      fetch_synapses, SynapseCriteria,
                      merge_neuron_properties)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from scipy.stats import zscore

from scipy import spatial

"""
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python

"""
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

# start client
neuprint_client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0')

# %% Build connectivity matrix. Columns = LC types, Rows = postsynaptic types
LC_keys = ['LC4', 'LC6', 'LC9', 'LC11', 'LC12', 'LC15', 'LC16', 'LC18', 'LC21', 'LC26', 'LPLC1', 'LPLC2']

LC_to_gloms = pd.DataFrame()
LCLC_lobula = pd.DataFrame(data=np.zeros((len(LC_keys), len(LC_keys))), columns=LC_keys, index=LC_keys)

type_type_het = []
type_cell_het = []
for LC_source in LC_keys:
    # # # # # # # # # (1) Outputs in PVLP + PLP # # # # # # # # # # # # # # # # # #
    # search for outputs from this LC to something that is in PVLP, PLP
    # Note min_total_weight here applies to single LC to single postsynaptic cell, across all ROIs
    #       Filter out spurious or very rare connections
    to_gloms_neurons, to_gloms_connection = fetch_adjacencies(sources=NeuronCriteria(type=LC_source, regex=True, status='Traced'),
                                                                  # targets=NeuronCriteria(type='^((?!LC).)*$', regex=True, status='Traced'), # Connects to non-LC cells
                                                                  # targets=NeuronCriteria(type='LPLC[0-9].*|LC[0-9].*', regex=True, status='Traced'),  # Connects to other LCs
                                                                  targets=NeuronCriteria(status='Traced'),  # Connects to any downstream cell
                                                                  rois=['PVLP(R)', 'PLP(R)'],
                                                                  min_total_weight=3)
    conn_df = merge_neuron_properties(to_gloms_neurons, to_gloms_connection, properties=['type', 'instance'])
    # Aggregate across individual cells. Group by cell type
    #       Sum total weights across individual pre/post-synaptic cells
    grouped = conn_df.groupby(['type_post']).sum()['weight']

    # make a new dataframe for this LC type and merge it to the whole LC-pop dataframe
    new_df = pd.DataFrame(data=grouped.values, columns=[LC_source], index=grouped.index.values)
    LC_to_gloms = pd.concat([LC_to_gloms, new_df], axis=1, sort=False)

    # Heterogeity across cells within an LC population
    grouped_post_cell = conn_df.groupby(['bodyId_pre', 'bodyId_post']).sum()['weight']
    grouped_post_type = conn_df.groupby(['bodyId_pre', 'type_post']).sum()['weight']

    het_type = pd.DataFrame()
    het_cell = pd.DataFrame()
    for bid in conn_df.bodyId_pre.unique():
        new_df = pd.DataFrame(data=grouped_post_type.get(bid).values, columns=[bid], index=grouped_post_type.get(bid).index.values)
        het_type = pd.concat([het_type, new_df], axis=1, sort=False)

        new_df = pd.DataFrame(data=grouped_post_cell.get(bid).values, columns=[bid], index=grouped_post_cell.get(bid).index.values)
        het_cell = pd.concat([het_cell, new_df], axis=1, sort=False)

    type_type_het.append(het_type)
    type_cell_het.append(het_cell)

    # # # # # # # # # (2) Outputs within LO # # # # # # # # # # # # # # # # # #
    to_lobula_neurons, to_lobula_connection = fetch_adjacencies(sources=NeuronCriteria(type=LC_source, regex=True, status='Traced'),
                                                                targets=NeuronCriteria(status='Traced'),  # Connects to any downstream cell
                                                                rois=['LO(R)'],
                                                                min_total_weight=3)
    conn_df = merge_neuron_properties(to_lobula_neurons, to_lobula_connection, properties=['type', 'instance'])
    grouped = conn_df.groupby('type_post').sum()['weight']
    lo_outputs = grouped.sum()
    for LC_target in LC_keys:
        if lo_outputs > 0:
            LCLC_lobula.loc[LC_source, LC_target] = grouped.get(LC_target, 0) / lo_outputs
        else:
            LCLC_lobula.loc[LC_source, LC_target] = 0

print('Found {} postsynaptic cells'.format(LC_to_gloms.shape[0]))
# %%


# %% Heterogeity across cells within an LC population

fh, ax = plt.subplots(2, 12, figsize=(16, 4))
for lc_ind, lc_type in enumerate(LC_keys):
    # Within an LC type - connections to downstream types
    show_map = type_type_het[lc_ind].sample(frac=1, axis=1).reset_index(drop=True)
    show_map[np.isnan(show_map)] = 0
    sns.heatmap(show_map, ax=ax[0, lc_ind], cbar=False)
    ax[0, lc_ind].set_title(lc_type)
    ax[0, lc_ind].set_xticks([])
    ax[0, lc_ind].set_yticks([])
    if lc_ind == 0:
        ax[0, lc_ind].set_ylabel('Across types')

    # Within an LC type - connections to downstream individual cells
    show_map = type_type_het[lc_ind].sample(frac=1, axis=1).reset_index(drop=True)
    show_map[np.isnan(show_map)] = 0
    sns.heatmap(show_map, ax=ax[1, lc_ind], cbar=False)
    ax[1, lc_ind].set_xticks([])
    ax[1, lc_ind].set_yticks([])
    if lc_ind == 0:
        ax[1, lc_ind].set_ylabel('Across cells')

# %%
for h in type_type_het:
    x = h.copy()
    x[np.isnan(x)] = 0
    sns.clustermap(x, row_cluster=False, col_cluster=True)



# %%
sns.heatmap(LC_to_gloms)

sns.heatmap(LCLC_lobula)

# %% Comparison to null connectivity model
# Threshold: total incoming synapses across an entire LC type to a single postsynaptic cell
thresh = 10

# Calculate P(connection) for each glom type
P_connection = (LC_to_gloms > thresh).sum(axis=0) / LC_to_gloms.shape[0]

observed_convergence = pd.DataFrame(data=0, index=LC_to_gloms.columns, columns=LC_to_gloms.columns)
null_convergence = pd.DataFrame(data=0, index=LC_to_gloms.columns, columns=LC_to_gloms.columns)
for g1 in LC_to_gloms.columns:
    for g2 in LC_to_gloms.columns:
        if g1 != g2:
            # observed convergent pairs
            in1 = LC_to_gloms[g1]
            in2 = LC_to_gloms[g2]
            convergents = np.logical_and(in1 > thresh, in2 > thresh)
            observed_convergence.loc[g1, g2] = convergents.sum()

            # Null model
            null_convergence.loc[g1, g2] = (P_connection[g1] * P_connection[g2]) * LC_to_gloms.shape[0]

fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(null_convergence, ax=ax[0], vmin=0, vmax=30)
ax[0].set_title('Null model')
sns.heatmap(observed_convergence, ax=ax[1], vmin=0, vmax=30)
ax[1].set_title('Observed')

fh, ax = plt.subplots(1, 1, figsize=(4, 3))
sns.heatmap(observed_convergence - null_convergence, ax=ax, cmap='RdBu', vmin=-20, vmax=20)
print('Expected {} convergent connections'.format(np.int(null_convergence.to_numpy()[np.triu_indices(12)].sum())))
print('Observed {} convergent connections'.format(observed_convergence.to_numpy()[np.triu_indices(12)].sum()))
# %%
clusters = {'A': ['LC26', 'LC6', 'LC12', 'LC15'],
            'B': ['LC9', 'LC11', 'LC18', 'LC21'],
            'C': ['LPLC1', 'LC16', 'LPLC2']}

clust_convergence_observed = pd.DataFrame(data=0, columns=clusters.keys(), index=clusters.keys())
clust_convergence_null = pd.DataFrame(data=0, columns=clusters.keys(), index=clusters.keys())
for clust_1 in list(clusters.keys()):
    for clust_2 in list(clusters.keys()):

        AB_observed = 0
        AB_null = 0
        for glom_1 in clusters.get(clust_1):
            for glom_2 in clusters.get(clust_2):
                AB_observed += observed_convergence.loc[glom_1, glom_2]
                AB_null += null_convergence.loc[glom_1, glom_2]

        clust_convergence_observed.loc[clust_1, clust_2] = AB_observed
        clust_convergence_null.loc[clust_1, clust_2] = np.int(AB_null)

print('Observed:')
print(clust_convergence_observed)
print('Null model:')
print(clust_convergence_null)

fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(clust_convergence_null, ax=ax[0], vmin=0, vmax=99)
ax[0].set_title('Null model')
sns.heatmap(clust_convergence_observed, ax=ax[1], vmin=0, vmax=99)
ax[1].set_title('Observed')


# %% Euclidean distance matrix in projection space
#   Axes are connectios to each postsynaptic cell type. Points are LC classes
X_conn = LC_to_gloms.copy().to_numpy().T
X_conn[np.isnan(X_conn)] = 0
X_conn = zscore(X_conn, axis=1)
D_conn = spatial.distance_matrix(X_conn, X_conn)

X_fxn = pd.read_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl')).to_numpy()
X_fxn = zscore(X_fxn, axis=1)
D_fxn = spatial.distance_matrix(X_fxn, X_fxn)

X_peaks = np.load(os.path.join(save_directory, 'pgs_responsepeaks.npy')).mean(axis=-1)  # glom x stim (avg over flies)
X_peaks = zscore(X_peaks, axis=1)
D_peaks = spatial.distance_matrix(X_peaks, X_peaks)

fh, ax = plt.subplots(1, 3, figsize=(12, 4))
sns.heatmap(D_conn, ax=ax[0])
sns.heatmap(D_fxn, ax=ax[1])
ax[2].plot(D_conn[np.triu_indices(12, k=1)], D_fxn[np.triu_indices(12, k=1)], 'bo')
# np.corrcoef(D_conn[np.triu_indices(12, k=1)], D_fxn[np.triu_indices(12, k=1)])
#
#
# fh, ax = plt.subplots(1, 3, figsize=(12, 4))
# sns.heatmap(D_conn, ax=ax[0])
# sns.heatmap(D_peaks, ax=ax[1])
# ax[2].plot(D_conn[np.triu_indices(12, k=1)], D_peaks[np.triu_indices(12, k=1)], 'bo')
# # np.corrcoef(D_conn[np.triu_indices(12, k=1)], D_peaks[np.triu_indices(12, k=1)])




# %% Projection space. What LCs are close together in projection space?
# Dimensionality of projection space is n postsynaptic cells
# X is (samples x features) = (LC x postsynaptic cell)

X = LC_to_gloms.copy().T  # LC x postsynaptic cell
# Shift each feature (column) to have 0 mean and unit variance
X[np.isnan(X)] = 0
X = zscore(X, axis=0)
# X = RobustScaler().fit_transform(X)
print('X = {} (samples, features)'.format(X.shape))

pca = PCA(n_components=12)
pca.fit(X)

fh4, ax4 = plt.subplots(1, 1, figsize=(3, 3))
ax4.plot(pca.explained_variance_ratio_, 'ko')
ax4.set_xlabel('Mode')
ax4.set_ylabel('Frac. var. explained')

fh5, ax5 = plt.subplots(3, 1, figsize=(4, 3))
ax5[0].plot(pca.components_[0, :], 'r-')
ax5[1].plot(pca.components_[1, :], 'g-')
ax5[2].plot(pca.components_[2, :], 'b-')

x_r = pca.fit_transform(X)  # cols = modes
fh6, ax6 = plt.subplots(1, 1, figsize=(6, 6))
ax6.scatter(x_r[:, 0], x_r[:, 1])
for lc_ind, lc_name in enumerate(LC_keys):
    ax6.annotate(lc_name, (x_r[lc_ind, 0], x_r[lc_ind, 1]))




# %%

tmp = LC_to_gloms.iloc[np.where(np.isnan(LC_to_gloms).sum(axis=1) < 11)]
conv_corr = tmp.corr()

fxnal_corr = pd.read_pickle(os.path.join(save_directory, 'pgs_corrmat.pkl'))


fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(conv_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[0])
sns.heatmap(fxnal_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[1])


# %%

plt.plot(conv_corr.to_numpy()[np.triu_indices(12, k=1)], fxnal_corr.to_numpy()[np.triu_indices(12, k=1)], 'bo')

# %%
fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(LCLC_lobula, cmap='RdBu_r', ax=ax[0])
sns.heatmap(fxnal_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[1])

plt.plot(LCLC_lobula.to_numpy()[np.triu_indices(12, k=1)], fxnal_corr.to_numpy()[np.triu_indices(12, k=1)], 'bo')

np.triu_indices(12, k=1)[0].shape
# %%
tmp = LC_to_gloms.copy().T.to_numpy()
tmp[np.isnan(tmp)] = 0
tmp = zscore(tmp, axis=1)

print(tmp.std(axis=1))

df = pd.DataFrame(data=tmp, index=LC_to_gloms.columns, columns=LC_to_gloms.index)
sns.clustermap(df, col_cluster=False, row_cluster=True,
               figsize=(8, 6))
# %%
response_mat = pd.read_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl'))
tmp = response_mat.copy().to_numpy()
tmp = zscore(tmp, axis=1)
print(tmp.std(axis=1))

df = pd.DataFrame(data=tmp, index=response_mat.index, columns=response_mat.columns)
sns.clustermap(response_mat, col_cluster=False, row_cluster=True, z_score=0, figsize=(8, 6))

# %%

tmp = LC_to_gloms.copy()
tmp[np.isnan(tmp)] = 0
tmp = zscore(tmp, axis=1)

df = pd.DataFrame(data=tmp, index=LC_to_gloms.index, columns=LC_to_gloms.columns)
sns.clustermap(df, col_cluster=False, row_cluster=True, figsize=(8, 6))


# %% Which LCs project to descending neurons?

DN_inds = np.where(['DN' in x or x=='Giant Fiber' for x in LC_to_gloms.index.values])[0]
# print(LC_to_gloms.index.values[DN_inds])

sns.heatmap(LC_to_gloms.iloc[DN_inds, :])
# %%
# Types:
#   OA: octopamine
#   AVLP - 160 types
#   CL: Clamp
#   PLP - 51
#   PVLP - 136
#   WED, SAD & AMMC-A1: gets LC4, LPLC1,2 inputs
#   SLP & SMP


type_inds = np.where(['PVLP' in x for x in LC_to_gloms.index.values])[0]
print(LC_to_gloms.index.values[type_inds])
print(len(type_inds))
if len(type_inds)>0:
    sns.heatmap(LC_to_gloms.iloc[type_inds, :])


# %%

to_lobula_neurons, to_lobula_connection = fetch_adjacencies(sources=NeuronCriteria(type=LC_source, regex=True, status='Traced'),
                                                            targets=NeuronCriteria(status='Traced'),  # Connects to any downstream cell
                                                            rois=['LO(R)'],
                                                            min_total_weight=3)

to_lobula_neurons
# %%
