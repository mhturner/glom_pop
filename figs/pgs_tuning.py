from visanalysis.analysis import imaging_data, volumetric_data
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
series_number = 1

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = volumetric_data.VolumetricDataObject(file_path,
                                          series_number,
                                          quiet=True)

# Load response data
response_data = dataio.loadResponses(ID, response_set_name='glom_20210809')

vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

meanbrain_red = response_data.get('meanbrain')[..., 0]
meanbrain_green = response_data.get('meanbrain')[..., 1]

# Align responses
mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(response_data.get('epoch_response'))
n_stimuli = mean_voxel_response.shape[2]
concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1) # responses, time (concat stims)
# %% GLOM MAP

z_to_show = [2, 4, 6, 8]

cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(response_data.get('mask') == 0, response_data.get('mask'))  # mask at 0

fh, ax = plt.subplots(len(z_to_show), 2, figsize=(4, 8))
# [x.set_xticklabels([]) for x in ax.ravel()]
# [x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

for z_ind, z in enumerate(z_to_show):
    ax[z_ind, 0].imshow(meanbrain_red[:, :, z].T, cmap='Reds')
    ax[z_ind, 1].imshow(glom_tmp[:, :, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind == 0:
        ax[z_ind, 0].set_title('mtdTomato')
        ax[z_ind, 1].set_title('Glomerulus map')

        dx = 25 / np.float(ID.getAcquisitionMetadata().get('micronsPerPixel_XAxis'))  # um -> pix
        ax[z_ind, 0].plot([17, 17+dx], [90, 90], color='k', linestyle='-', marker='None', linewidth=2)
        ax[z_ind, 0].annotate('25 um', (18, 87), color='k')

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.set_xlim([15, 120])
    x.set_ylim([95, 5])

handles = [Patch(facecolor=color) for color in colors]
fh.legend(handles, [label for label in names], fontsize=8, ncol=4, handleheight=1.0, labelspacing=0.05)


# %% Glom responses
fh, ax = plt.subplots(concatenated_tuning.shape[0], 1, figsize=(14, 14))
[util.cleanAxes(x) for x in ax]
[x.set_ylim([-0.25, 0.5])]
for g_ind, name in enumerate(names):
    ax[g_ind].plot(concatenated_tuning[g_ind], color=colors[g_ind, :])
    ax[g_ind].set_ylabel(name)


# %% Compare gloms across animals

series_numbers = [1, 4, 7]

resps = []
for series_number in series_numbers:
    experiment_file_name = '2021-08-04'

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

    # ImagingDataObject wants a path to an hdf5 file and a series number from that file
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom_20210809')

    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    meanbrain_red = response_data.get('meanbrain')[..., 0]
    meanbrain_green = response_data.get('meanbrain')[..., 1]

    # Align responses
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(response_data.get('epoch_response'))
    n_stimuli = mean_voxel_response.shape[2]
    concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1) # responses, time (concat stims)

    resps.append(concatenated_tuning)

resps = np.dstack(resps)
# %%

within_fly = []
for f_ind in range(len(series_numbers)):
    cmat = pd.DataFrame(data=resps[:, :, f_ind].T).corr()
    corr_vals = cmat.to_numpy()[np.triu_indices(cmat.shape[0], k=1)]
    within_fly.append(corr_vals)

between_fly = []
for g_ind, name in enumerate(names):
    cmat = pd.DataFrame(data=resps[g_ind, :, :]).corr()
    corr_vals = cmat.to_numpy()[np.triu_indices(len(series_numbers), k=1)]
    between_fly.append(corr_vals)

# %%
fh, ax = plt.subplots(1, 1, figsize=(9, 4))
for g_ind, name in enumerate(names):
    ax.plot(g_ind*np.ones(3), between_fly[g_ind], 'kx')

ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names);
ax.set_ylim([-0.2, 1.0])
ax.axhline(color='k', alpha=0.5)

within_fly_mean = np.median(np.stack(within_fly).ravel())
ax.axhline(within_fly_mean, color='b', alpha=0.5)

# %%

np.stack(between_fly).ravel().shape

np.stack(within_fly).ravel().shape
# %%

h, p = stats.ttest_ind(np.stack(within_fly).ravel(), np.stack(between_fly).ravel())
h
p

np.median(np.stack(within_fly).ravel())
np.median(np.stack(between_fly).ravel())
