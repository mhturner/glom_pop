from visanalysis.analysis import imaging_data, volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
from scipy import stats

from glom_pop import dataio, util


experiment_file_directory = '/Users/mhturner/CurrentData'
experiment_file_name = '2021-08-04'
series_number = 7  # 1, 4, 7

save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = volumetric_data.VolumetricDataObject(file_path,
                                          series_number,
                                          quiet=True)

# Load response data
response_data = dataio.loadResponses(ID, response_set_name='glom_20210817', get_voxel_responses=False)

vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

meanbrain_red = response_data.get('meanbrain')[..., 0]
meanbrain_green = response_data.get('meanbrain')[..., 1]

# Align responses
mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(response_data.get('epoch_response'))
n_stimuli = mean_voxel_response.shape[2]
concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)


# %% GLOM MAP

# z_to_show = [2, 4, 6, 8, 10]
z_to_show = [1, 3, 5, 7, 9]

cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(response_data.get('mask') == 0, response_data.get('mask'))  # mask at 0

fh, ax = plt.subplots(2, len(z_to_show), figsize=(16, 4))
[x.set_xticklabels([]) for x in ax.ravel()]
[x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

for z_ind, z in enumerate(z_to_show):
    ax[0, z_ind].imshow(meanbrain_red[:, :, z].T, cmap='Reds')
    ax[1, z_ind].imshow(glom_tmp[:, :, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind == 0:
        ax[0, z_ind].set_title('mtdTomato')
        ax[1, z_ind].set_title('Glomerulus map')

        dx = 25 / np.float(ID.getAcquisitionMetadata().get('micronsPerPixel_XAxis'))  # um -> pix
        ax[0, z_ind].plot([17, 17+dx], [90, 90], color='k', linestyle='-', marker='None', linewidth=2)
        ax[0, z_ind].annotate('25 um', (16, 87), color='k')

for x in ax.ravel():
    x.locator_params(axis='y', nbins=6)
    x.locator_params(axis='x', nbins=6)
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.set_xlim([15, 120])
    x.set_ylim([95, 5])

handles = [Patch(facecolor=color) for color in colors]
fh.legend(handles, [label for label in names], fontsize=8, ncol=4, handleheight=1.0, labelspacing=0.05)
fh.savefig(os.path.join(save_directory, 'glom_overlay_{}_{}.pdf'.format(experiment_file_name, series_number)))

# %% Glom responses

fh, ax = plt.subplots(1 + concatenated_tuning.shape[0], len(unique_parameter_values), figsize=(18, 18))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 0.75]) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.05, hspace=0.05)

for u_ind, un in enumerate(unique_parameter_values):

    params = {'center': [0, 0]}
    if un[0] == 'MovingRectangle':
        params['width'] = 10
        params['height'] = 50
        params['color'] = un[2] * np.ones(3)
        params['direction'] = un[1]
    elif un[0] == 'ExpandingMovingSpot':
        params['radius'] = un[1] / 2
        params['color'] = un[2] * np.ones(3)
        if un[3] < 0:
            params['direction'] = 180
        elif un[3] > 0:
            params['direction'] = 0

    plot_tools.addStimulusDrawing(ax[0, u_ind], stimulus=un[0], params=params)

    for g_ind, name in enumerate(names):
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[g_ind, :, u_ind], color=colors[g_ind, :])
        ax[g_ind+1, u_ind].axhline(color='k', alpha=0.5)
        if (g_ind == 0) & (u_ind == 0):
            plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=0, F_value=-0.2)

        if u_ind == 0:
            ax[g_ind+1, u_ind].set_ylabel(name)

fh.savefig(os.path.join(save_directory, 'ind_tuning_{}_{}.pdf'.format(experiment_file_name, series_number)))




# %%

series = [
          ('2021-08-04', 1),
          ('2021-08-04', 4),
          ('2021-08-04', 7),
          ('2021-08-11', 1),
          # ('2021-08-11', 4),  # Not very responsive gloms, see note in .h5
          ('2021-08-11', 7),
          ]


fh, ax = plt.subplots(1 + concatenated_tuning.shape[0], len(unique_parameter_values), figsize=(18, 18))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 1.0]) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.05, hspace=0.05)

all_responses = []
for s_ind, ser in enumerate(series):
    experiment_file_name = ser[0]
    series_number = ser[1]

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom_20210817')
    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(response_data.get('epoch_response'))
    n_stimuli = mean_voxel_response.shape[2]

    all_responses.append(mean_voxel_response)

    for u_ind, un in enumerate(unique_parameter_values):
        if s_ind == 0:
            params = {'center': [0, 0]}
            if un[0] == 'MovingRectangle':
                params['width'] = 10
                params['height'] = 50
                params['color'] = un[2] * np.ones(3)
                params['direction'] = un[1]
            elif un[0] == 'ExpandingMovingSpot':
                params['radius'] = un[1] / 2
                params['color'] = un[2] * np.ones(3)
                if un[3] < 0:
                    params['direction'] = 180
                elif un[3] > 0:
                    params['direction'] = 0

            plot_tools.addStimulusDrawing(ax[0, u_ind], stimulus=un[0], params=params)

        for g_ind, name in enumerate(names):
            ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[g_ind, :, u_ind], color=colors[g_ind, :], alpha=0.25)
            # ax[g_ind+1, u_ind].axhline(color='k', alpha=0.5)
            if (g_ind == 0) & (u_ind == 0) & (s_ind==0):
                plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=0, F_value=-0.2)

            if (u_ind == 0) & (s_ind==0):
                ax[g_ind+1, u_ind].set_ylabel(name)


all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
mean_responses = np.mean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.std(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)

for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(names):
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_responses[g_ind, :, u_ind], color=colors[g_ind, :], alpha=1.0, linewidth=2)

fh.savefig(os.path.join(save_directory, 'mean_tuning.pdf'))
# %%

import umap

# %%

# %%
tmp.shape

tmp = np.concatenate([all_responses[:, :, x, :] for x in range(n_stimuli)], axis=1)
all_glom_responses = np.concatenate([tmp[:, :, x] for x in range(len(series))], axis=0)  # total gloms (across flies), concatenated time
all_glom_responses.shape

glom_ids = np.tile(np.arange(0, tmp.shape[0]), len(series))
# %%
from sklearn.decomposition import PCA, SparsePCA

pca = PCA(svd_solver='full')

pca.fit(all_glom_responses)
pca.components_.shape


# %%
pca = SparsePCA()
pca.fit(all_glom_responses)

reduced_data = SparsePCA(n_components=2).fit_transform(all_glom_responses)
reduced_data.shape
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=glom_ids, cmap=cc.cm.glasbey)


# %%
plt.plot(pca.components_[0,:])

pca.explained_variance_ratio_.shape
plt.plot(pca.explained_variance_ratio_, 'ko')

pca.singular_values_.shape


reduced_data = PCA(n_components=2).fit_transform(all_glom_responses)

reduced_data.shape
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=glom_ids, cmap=cc.cm.glasbey)
# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(all_glom_responses)


embedding.shape
plt.scatter(embedding[:, 0], embedding[:, 1], c=glom_ids, cmap=cc.cm.glasbey)

# %%
