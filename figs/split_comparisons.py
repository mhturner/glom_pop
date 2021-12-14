import numpy as np
import matplotlib.pyplot as plt
import os
import colorcet as cc

from glom_pop import dataio, util
from visanalysis.analysis.shared_analysis import filterDataFiles
from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

# TODO: time resampling for early LC11 data...
# target_gloms = ['LC4', 'LC9', 'LC11', 'LC18']
# target_gloms = ['LC4', 'LC9', 'LC18']
target_gloms = ['LC9']

mean_chat_responses = np.load(os.path.join(save_directory, 'mean_chat_responses.npy'))
included_gloms = np.load(os.path.join(save_directory, 'included_gloms.npy'))
colors = np.load(os.path.join(save_directory, 'colors.npy'))

fh, ax = plt.subplots(len(target_gloms)+1, 32, figsize=(18, 16))
# [x.set_axis_off() for x in ax.ravel()]
[plot_tools.cleanAxes(x) for x in ax.ravel()]

[x.set_ylim([-0.2, 1.2]) for x in ax.ravel()]
fh.subplots_adjust(wspace=0.00, hspace=0.00)
for g_ind, target_glom in enumerate(target_gloms):
    chat_glom_ind = np.where(included_gloms == target_glom)[0][0]

    target_series = filterDataFiles(experiment_file_directory,
                                    target_fly_metadata={'driver_1': target_glom},
                                    target_series_metadata={'protocol_ID': 'PanGlomSuite'},
                                    target_roi_series=['glom'])

    split_responses = []
    for s_ind, ser in enumerate(target_series):
        file_path = ser.get('file_name') + '.hdf5'
        series_number = ser.get('series')
        ID = volumetric_data.VolumetricDataObject(file_path,
                                                  series_number,
                                                  quiet=True)

        # Align responses
        time_vector, voxel_trial_matrix = ID.getTrialAlignedVoxelResponses(ID.getRoiResponses('glom').get('roi_response')[0], dff=True)
        mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(voxel_trial_matrix)
        n_stimuli = mean_voxel_response.shape[2]

        split_responses.append(mean_voxel_response)

    split_responses = np.vstack(split_responses)
    for u_ind, un in enumerate(unique_parameter_values[:-2]):
        ax[g_ind, u_ind].plot(time_vector, split_responses.mean(axis=0)[:, u_ind], color='k', alpha=0.5, linewidth=2)
        # ax[g_ind, u_ind].plot(time_vector, np.squeeze(split_responses[1, :, u_ind]).T, alpha=0.5, linewidth=2, color='k')

        ax[g_ind, u_ind].plot(time_vector, mean_chat_responses[chat_glom_ind, :, u_ind], color=colors[chat_glom_ind, :], alpha=0.5, linewidth=2)

    ax[g_ind, 0].annotate(target_glom, (0, 0.25), rotation=90)

plot_tools.addScaleBars(ax[0, 0], dT=2, dF=0.50, T_value=0, F_value=-0.08)
fh.legend()


# %%

mean_chat_responses.shape

# %%
