from neuprint import (Client, fetch_adjacencies, NeuronCriteria, fetch_neurons, fetch_primary_rois,
                      merge_neuron_properties)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
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

# %%
# # # # # # BUILD CONNECTIVITY MATRIX # # # # # # # # # # # # # # # # # # # # #
#  Columns = LC types, Rows = postsynaptic types
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
fh0, ax0 = plt.subplots(1, 1, figsize=(5, 4))
sns.heatmap(LC_to_gloms, ax=ax0, cbar_kws={'label': 'Total synapses'})
ax0.set_xlabel('Presynaptic')
ax0.set_ylabel('Postsynaptic')
fh0.savefig(os.path.join(save_directory, 'downstream_heatmap.png'), bbox_inches='tight', dpi=350)
# %%

# %%

#
# tt = LC_to_gloms.index.values[0]
# tt
# neur, conn = fetch_neurons(NeuronCriteria(type=tt))
# output_rois = pd.DataFrame(columns=all_rois)
# for cell in range(neur.shape[0]):
#     for key in neur.roiInfo[0]:
#         ds = neur.roiInfo[0].get(key).get('downstream', 0)
#

# %%
# # # # # # HETEROGENEITY ACROSS CELLS WITHIN AN LC POPULATION # # # # # # # # # # # # # # # # # # # # #

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
# # # # # # CONVERGENCE: COMPARE TO NULL CONNECTIVITY MODEL # # # # # # # # # # # # # # # # # # # # #


def computeConvergence(connectivity):
    convergence = pd.DataFrame(data=0, index=connectivity.columns, columns=connectivity.columns)
    for g1 in connectivity.columns:
        for g2 in connectivity.columns:
            if g1 != g2:
                convergence.loc[g1, g2] = np.logical_and(connectivity[g1], connectivity[g2]).sum()

    return convergence


# Threshold: total incoming synapses across an entire LC type to a single postsynaptic cell
# Thresh (for each LC class) = mean connection from LC to postsyn cell
thresh = LC_to_gloms.mean()

# Binarized observed connectivity matrix
observed_connectivity = LC_to_gloms >= thresh
observed_total_connections = observed_connectivity.to_numpy().ravel().sum()
observed_convergence = computeConvergence(observed_connectivity)

observed_convergence
# Convergent downstream connections, binary matrix
observed_downstream = observed_connectivity.loc[observed_connectivity.sum(axis=1) >= 2, :]

# Make the null connectivity matrix model
# Calculate P(connection) for each glom type
P_connection = observed_connectivity.sum(axis=0) / observed_connectivity.shape[0]

# Iterate over null model connectivity matrices
null_convergence = []
null_total_connections = []
for it in range(100):
    null_connectivity = pd.DataFrame(data=np.nan, columns=LC_to_gloms.columns, index=LC_to_gloms.index)
    for x in null_connectivity.columns:
        null_connectivity[x] = np.random.choice([0, 1], size=null_connectivity.shape[0], p=[1-P_connection[x], P_connection[x]])
    null_convergence.append(computeConvergence(null_connectivity).to_numpy())
    null_total_connections.append(null_connectivity.to_numpy().ravel().sum())

null_convergence = np.dstack(null_convergence)
null_convergence_mean = pd.DataFrame(data=null_convergence.mean(axis=-1), columns=LC_to_gloms.columns, index=LC_to_gloms.columns)

print('Observed total connnections = {}'.format(observed_total_connections))
print('Null total connections = {} +/- {}'.format(np.mean(null_total_connections), np.std(null_total_connections)))

# %%

vmax = np.max([null_convergence_mean.to_numpy().ravel().max(), observed_convergence.to_numpy().ravel().max()])
fh1, ax1 = plt.subplots(1, 3, figsize=(11, 3))
sns.heatmap(null_convergence_mean, ax=ax1[0], vmin=0, vmax=vmax)
ax1[0].set_title('Null model')
sns.heatmap(observed_convergence, ax=ax1[1], vmin=0, vmax=vmax)
ax1[1].set_title('Observed')

diff = observed_convergence - null_convergence_mean
v_lim = np.ceil(np.abs(diff.to_numpy().ravel()).max())
sns.heatmap(diff, ax=ax1[2], cmap='RdBu_r', vmin=-v_lim, vmax=+v_lim)
ax1[2].set_title('Observed - Null')
print('Expected {} convergent connections'.format(np.int(null_convergence_mean.to_numpy()[np.triu_indices(null_convergence_mean.shape[0])].sum())))
print('Observed {} convergent connections'.format(observed_convergence.to_numpy()[np.triu_indices(null_convergence_mean.shape[0])].sum()))

fh2, ax2 = plt.subplots(1, 1, figsize=(4, 6))
sns.heatmap(observed_downstream, ax=ax2, cmap='Greys')

fh1.savefig(os.path.join(save_directory, 'downstream_conv_model.png'), bbox_inches='tight', dpi=350)
fh2.savefig(os.path.join(save_directory, 'downstream_conv_observed.png'), bbox_inches='tight', dpi=350)

# %%

all_rois = fetch_primary_rois()
output_rois = pd.DataFrame(data=0, columns=all_rois, index=observed_downstream.index.values)

for downstream in output_rois.index.values:
    neur, conn = fetch_neurons(NeuronCriteria(type=downstream))
    for cell in range(neur.shape[0]):
        for key in neur.roiInfo[cell]:
            ds = neur.roiInfo[cell].get(key).get('downstream', 0)
            if key in all_rois:
                output_rois.loc[downstream, key] += ds


# %%
fh, ax = plt.subplots(1, 1, figsize=(15, 4))

sns.heatmap(output_rois, ax=ax)
ax.set_xlabel('Output roi')

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
                AB_null += null_convergence_mean.loc[glom_1, glom_2]

        clust_convergence_observed.loc[clust_1, clust_2] = AB_observed
        clust_convergence_null.loc[clust_1, clust_2] = np.int(AB_null)

print('Observed:')
print(clust_convergence_observed)
print('Null model:')
print(clust_convergence_null)

vmax = np.max([clust_convergence_null.to_numpy().ravel().max(), clust_convergence_observed.to_numpy().ravel().max()])
fh, ax = plt.subplots(1, 3, figsize=(10, 3))
sns.heatmap(clust_convergence_null, ax=ax[0], vmin=0, vmax=vmax, cmap='Greys')
ax[0].set_title('Null model')
sns.heatmap(clust_convergence_observed, ax=ax[1], vmin=0, vmax=vmax, cmap='Greys')
ax[1].set_title('Observed')

diff = clust_convergence_observed - clust_convergence_null
v_lim = np.ceil(np.abs(diff.to_numpy().ravel()).max())
sns.heatmap(diff, ax=ax[2], vmin=-v_lim, vmax=v_lim, cmap='RdBu_r')
ax[2].set_title('Observed-Null')


# %%
# # # # # # DISTANCE IN PROJECTION SPACE # # # # # # # # # # # # # # # # # # # # #
#   Axes are connections to each postsynaptic cell type. Points are LC classes
X_conn = LC_to_gloms.copy().to_numpy().T
X_conn[np.isnan(X_conn)] = 0
X_conn = zscore(X_conn, axis=1)
D_conn = spatial.distance_matrix(X_conn, X_conn)
np.fill_diagonal(D_conn, np.nan)
D_conn = pd.DataFrame(data=D_conn, index=LC_to_gloms.columns, columns=LC_to_gloms.columns)

X_fxn = pd.read_pickle(os.path.join(save_directory, 'pgs_responsemat.pkl')).to_numpy()
X_fxn = zscore(X_fxn, axis=1)
D_fxn = spatial.distance_matrix(X_fxn, X_fxn)
np.fill_diagonal(D_fxn, np.nan)
D_fxn = pd.DataFrame(data=D_fxn, index=LC_to_gloms.columns, columns=LC_to_gloms.columns)

fh, ax = plt.subplots(1, 3, figsize=(14, 3))
sns.heatmap(D_conn, ax=ax[0], cbar_kws={'label': 'Projection distance'})
sns.heatmap(D_fxn, ax=ax[1], cbar_kws={'label': 'Functional distance'})
ax[2].plot(D_conn.to_numpy()[np.triu_indices(D_conn.shape[0], k=1)], D_fxn.to_numpy()[np.triu_indices(D_fxn.shape[0], k=1)], 'ko')
ax[2].set_xlabel('Projection distance')
ax[2].set_ylabel('Functional distance')

r = np.corrcoef(D_conn.to_numpy()[np.triu_indices(D_conn.shape[0], k=1)], D_fxn.to_numpy()[np.triu_indices(D_fxn.shape[0], k=1)])[0, 1]
print('r = {}'.format(r))
# %% Projection space. What LCs are close together in projection space?
# Dimensionality of projection space is n postsynaptic cells
# X is (samples x features) = (LC x postsynaptic cell)

X = LC_to_gloms.copy().T  # LC x postsynaptic cell
# Shift each feature (column) to have 0 mean and unit variance
X[np.isnan(X)] = 0
X = zscore(X, axis=0)
# X = RobustScaler().fit_transform(X)
print('X = {} (samples, features)'.format(X.shape))

pca = PCA(n_components=LC_to_gloms.shape[1])
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




# %% Pairwise correlation in function and convergence space

tmp = LC_to_gloms.iloc[np.where(np.isnan(LC_to_gloms).sum(axis=1) < 11)]
conv_corr = tmp.corr().to_numpy()
conv_corr[np.isnan(conv_corr)] = 0
conv_corr = pd.DataFrame(data=conv_corr, index=tmp.columns, columns=tmp.columns)

fxnal_corr = pd.read_pickle(os.path.join(save_directory, 'pgs_corrmat.pkl'))

fh, ax = plt.subplots(1, 3, figsize=(14, 3))
sns.heatmap(conv_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[0])
ax[0].set_title('Convergent connection correlation')
sns.heatmap(fxnal_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[1])
ax[1].set_title('Functional tuning correlation')
ax[2].plot(conv_corr.to_numpy()[np.triu_indices(conv_corr.shape[0], k=1)], fxnal_corr.to_numpy()[np.triu_indices(fxnal_corr.shape[0], k=1)], 'ko')

r = np.corrcoef(conv_corr.to_numpy()[np.triu_indices(conv_corr.shape[0], k=1)], fxnal_corr.to_numpy()[np.triu_indices(fxnal_corr.shape[0], k=1)])[0, 1]
print('r = {}'.format(r))
# %% Within lobula connections vs. functional tuning
fh, ax = plt.subplots(1, 3, figsize=(14, 3))
sns.heatmap(LCLC_lobula, cmap='RdBu_r', ax=ax[0])
ax[0].set_title('Within lobula connections')
sns.heatmap(fxnal_corr, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax[1])
ax[1].set_title('Functional tuning correlation')

tmp = (LCLC_lobula.to_numpy() + LCLC_lobula.to_numpy().T) / 2
ax[2].plot(tmp[np.triu_indices(tmp.shape[0], k=1)], fxnal_corr.to_numpy()[np.triu_indices(fxnal_corr.shape[0], k=1)], 'bo')

# %% Cluster glomeruli by projection patterns
tmp = LC_to_gloms.copy().T.to_numpy()
tmp[np.isnan(tmp)] = 0
tmp = zscore(tmp, axis=1)


df = pd.DataFrame(data=tmp, index=LC_to_gloms.columns, columns=LC_to_gloms.index)
sns.clustermap(df, col_cluster=False, row_cluster=True,
               figsize=(8, 6))
# %% Cluster glomeruli by functional tuning
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

fh, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.heatmap(LC_to_gloms.iloc[DN_inds, :], ax=ax, cbar_kws={'label': 'Total synapses'})
ax.set_xlabel('Presynaptic')
ax.set_ylabel('Postsynaptic')


fh.savefig(os.path.join(save_directory, 'DN_inputs.png'), bbox_inches='tight')
# %%
# Types:
#   DN: Descending neurons
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
# # # # # # LOBULA INTRINSIC CONNECTIONS # # # # # # # # # # # # # # # # # # # # #
LC_keys = ['LC4', 'LC6', 'LC9', 'LC11', 'LC12', 'LC15', 'LC16', 'LC18', 'LC21', 'LC26', 'LPLC1', 'LPLC2']


from_li_neurons, from_li_connection = fetch_adjacencies(sources=NeuronCriteria(type='Li[0-9].*', regex=True, status='Traced'),
                                                        targets=NeuronCriteria(type='LPLC[0-9].*|LC[0-9].*', regex=True, status='Traced'),  # Connects to any downstream cell
                                                        min_total_weight=3)
from_li = merge_neuron_properties(from_li_neurons, from_li_connection, properties=['type', 'instance'])

to_li_neurons, to_li_connection = fetch_adjacencies(sources=NeuronCriteria(type='LPLC[0-9].*|LC[0-9].*', regex=True, status='Traced'),
                                                    targets=NeuronCriteria(type='Li[0-9].*', regex=True, status='Traced'),  # connects to Li cell
                                                    min_total_weight=3)
to_li = merge_neuron_properties(to_li_neurons, to_li_connection, properties=['type', 'instance'])

neur, _ = fetch_neurons(NeuronCriteria(type='Li[0-9].*', regex=True, status='Traced'))
Li_types = np.sort(neur.type.unique())

Li_inputs = pd.DataFrame(data=0, index=LC_keys, columns=Li_types)
Li_outputs = pd.DataFrame(data=0, index=Li_types, columns=LC_keys)
for Li in Li_types:
    tmp_to = to_li.iloc[np.where(to_li.type_post.values==Li)[0], :]
    inputs = tmp_to.groupby('type_pre').sum()['weight']
    Li_inputs.loc[:, Li] = inputs

    tmp_from = from_li.iloc[np.where(from_li.type_pre.values==Li)[0], :]
    outputs = tmp_from.groupby('type_post').sum()['weight']
    Li_outputs.loc[Li, :] = outputs


# %%


fh, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(Li_inputs, ax=ax[0])
sns.heatmap(Li_outputs, ax=ax[1])
# %%
