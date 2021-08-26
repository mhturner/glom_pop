from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc

from glom_pop import dataio, util

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'


# %% PLOT MEAN + INDIVIDUAL RESPONSES TO TUNING SUITE

series = [
          ('2021-08-04', 1),
          ('2021-08-04', 4),
          # ('2021-08-04', 7),  # somewhat weak responses
          ('2021-08-11', 1),
          # ('2021-08-11', 4),  # Not very responsive gloms, see note in .h5
          ('2021-08-11', 7),
          ('2021-08-20', 2),
          ('2021-08-20', 6),  # Check moco on this?
          ]


fh, ax = plt.subplots(1 + 14, 32, figsize=(18, 18))
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
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)
    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))
    cmap = cc.cm.glasbey
    colors = cmap(vals/vals.max())

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
            if (g_ind == 0) & (u_ind == 0) & (s_ind == 0):
                plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=0, F_value=-0.2)

            if (u_ind == 0) & (s_ind == 0):
                ax[g_ind+1, u_ind].set_ylabel(name)


all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, time, param, fly)
mean_responses = np.mean(all_responses, axis=-1)  # (glom, time, param)
sem_responses = np.std(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, time, param)

for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(names):
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_responses[g_ind, :, u_ind], color=colors[g_ind, :], alpha=1.0, linewidth=2)

fh.savefig(os.path.join(save_directory, 'mean_tuning.pdf'))
