"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import umap


from glom_pop import dataio, util


experiment_file_directory = '/Users/mhturner/CurrentData'
experiment_file_name = '2021-08-04'
series_number = 1  # 1, 4, 7

save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = volumetric_data.VolumetricDataObject(file_path,
                                          series_number,
                                          quiet=True)

# Load response data
response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=True)

vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

meanbrain_red = response_data.get('meanbrain')[..., 0]
meanbrain_green = response_data.get('meanbrain')[..., 1]

# Align responses
mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(response_data.get('epoch_response'))
n_stimuli = mean_voxel_response.shape[2]
concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)


# %% GLOM MAP

# z_to_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
z_to_show = [1, 3, 5, 7]

cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(response_data.get('mask') == 0, response_data.get('mask'))  # mask at 0

fh, ax = plt.subplots(2, len(z_to_show), figsize=(8, 4))
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

# %% GLOM RESPONSES TO TUNING SUITE

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



# %% UMAP EMBEDDING OF VOXEL RESPONSES, WITH GLOM MEMBERSHIP COLOR CODE

map_vals = list(response_data['voxel_epoch_responses'].keys())

all_glom_ids = []
all_voxel_responses = []
for mv in map_vals:
    erm = response_data['voxel_epoch_responses'][mv]
    # Align responses
    mean_voxel_response, _, _, _, _, _ = ID.getMeanBrainByStimulus(erm)
    n_stimuli = mean_voxel_response.shape[2]
    concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)

    all_voxel_responses.append(concatenated_tuning)
    all_glom_ids.append(mv*np.ones(concatenated_tuning.shape[0]))

all_voxel_responses = np.vstack(all_voxel_responses)
all_glom_ids = np.hstack(all_glom_ids)

reducer = umap.UMAP()
embedding = reducer.fit_transform(all_voxel_responses)

# %% Plot UMAP embedding of each glom's voxels

fh, ax = plt.subplots(2, 7, figsize=(9, 3.5))
ax = ax.ravel()
for m_ind, mv in enumerate(map_vals):
    ax[m_ind].scatter(embedding[:, 0], embedding[:, 1], color=[0.5, 0.5, 0.5], alpha=0.5, marker='.')
    ax[m_ind].scatter(embedding[mv == all_glom_ids, 0], embedding[mv == all_glom_ids, 1], color=colors[m_ind], marker='.')
    # ax[m_ind].set_title(names.values[m_ind])
    ax[m_ind].annotate(names.values[m_ind], (embedding[:, 0].min(), 0.9*embedding[:, 1].max()))
    if m_ind == 0:
        ax[m_ind].set_xticks([])
        ax[m_ind].set_yticks([])
        ax[m_ind].set_xlabel('UMAP Dim. 1')
        ax[m_ind].set_ylabel('UMAP Dim. 2')
        ax[m_ind].spines['top'].set_visible(False)
        ax[m_ind].spines['right'].set_visible(False)
    else:
        ax[m_ind].set_axis_off()


fh.savefig(os.path.join(save_directory, 'embedding_{}_{}.pdf'.format(experiment_file_name, series_number)))


# %%