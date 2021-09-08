from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
import pandas as pd

from glom_pop import dataio, util

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'


# %% PLOT MEAN + INDIVIDUAL RESPONSES TO TUNING SUITE

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

fh, ax = plt.subplots(1 + len(included_gloms), 32, figsize=(7, 8.5))
[util.cleanAxes(x) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.00, hspace=0.00)

all_responses = []
vox_per_glom = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)
    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))
    cmap = cc.cm.glasbey
    colors = cmap(vals/vals.max())

    # voxels per glom
    vox_per_glom.append([np.sum(response_data.get('mask') == mv) for mv in vals])

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

        for g_ind, name in enumerate(included_gloms):
            pull_ind = np.where(name==names)[0][0]
            # ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[pull_ind, :, u_ind], color=colors[pull_ind, :], alpha=0.25)
            # ax[g_ind+1, u_ind].axhline(color='k', alpha=0.5)
            if (g_ind == 0) & (u_ind == 0) & (s_ind == 0):
                plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=-0.1, F_value=-0.1)

            if (u_ind == 0) & (s_ind == 0):
                ax[g_ind+1, u_ind].set_ylabel(name)


all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
mean_responses = np.mean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.std(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)
vox_per_glom = np.stack(vox_per_glom, axis=-1)

for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(included_gloms):
        pull_ind = np.where(name==names)[0][0]
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_responses[pull_ind, :, u_ind], color=colors[pull_ind, :], alpha=1.0, linewidth=2)
        ax[g_ind+1, u_ind].fill_between(response_data.get('time_vector'),
                                        mean_responses[pull_ind, :, u_ind] - sem_responses[pull_ind, :, u_ind],
                                        mean_responses[pull_ind, :, u_ind] + sem_responses[pull_ind, :, u_ind],
                                        color=colors[pull_ind, :], alpha=0.5, linewidth=0)


[x.set_ylim([mean_responses.min(), 0.8]) for x in ax.ravel()]

fh.savefig(os.path.join(save_directory, 'mean_tuning.pdf'))

# %%
response_amp.shape


# %%

pd.DataFrame(np.vstack([names.values, vox_per_glom.mean(axis=-1)]).T)
# %%
# series = [
#           ('2021-08-11', 10),  # R65B05
#           ]

series = [
          ('2021-08-20', 10),  # LC11
          ]

all_responses = []
for ser in series:
    experiment_file_name = ser[0]
    series_number = ser[1]
    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Align responses
    time_vector, voxel_trial_matrix = ID.getTrialAlignedVoxelResponses(ID.getRoiResponses('LC11').get('roi_response')[0], dff=True)
    mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(voxel_trial_matrix)
    n_stimuli = mean_voxel_response.shape[2]
    concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)

    all_responses.append(concatenated_tuning)

mean_responses.shape
response_amp

unique_parameter_values
# %%
concatenated_tuning.shape
plt.plot(concatenated_tuning.T)




# %%
