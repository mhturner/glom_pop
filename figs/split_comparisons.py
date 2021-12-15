import numpy as np
import matplotlib.pyplot as plt
import os
import colorcet as cc
import pandas as pd

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
target_gloms = ['LC4', 'LC9', 'LC18']
yoffset = 0.0
# target_gloms = ['LC9']

mean_chat_responses = np.load(os.path.join(save_directory, 'mean_chat_responses.npy'))
included_gloms = np.load(os.path.join(save_directory, 'included_gloms.npy'))
colors = np.load(os.path.join(save_directory, 'colors.npy'))

fh, ax = plt.subplots(len(target_gloms), 30, figsize=(18, 16))
[plot_tools.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.1, 1.2]) for x in ax.ravel()]

fh3, ax3 = plt.subplots(1, len(target_gloms), figsize=(6, 3))

for g_ind, target_glom in enumerate(target_gloms):
    chat_glom_ind = np.where(included_gloms == target_glom)[0][0]

    target_series = filterDataFiles(experiment_file_directory,
                                    target_fly_metadata={'driver_1': target_glom},
                                    target_series_metadata={'protocol_ID': 'PanGlomSuite'},
                                    target_roi_series=['glom'])

    fh2, ax2 = plt.subplots(len(target_series)+1, 1, figsize=(18, 16))
    # [plot_tools.cleanAxes(x) for x in ax2.ravel()]
    [x.set_ylim([-0.1, 0.6]) for x in ax2.ravel()]
    fh2.subplots_adjust(wspace=0.00, hspace=0.00)

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

        concat_resp = np.vstack(np.concatenate([mean_voxel_response[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))
        ax2[s_ind+1].plot(concat_resp.T)
        ax2[s_ind+1].set_ylabel('{}:{}'.format(ser.get('file_name').split('/')[-1], ser.get('series')))

    split_responses = np.vstack(split_responses)

    mean_concat = np.vstack(np.concatenate([split_responses.mean(axis=0)[:, x] for x in np.arange(len(unique_parameter_values))], axis=0))
    ax2[0].plot(mean_concat, color=colors[chat_glom_ind, :])

    # Compare mean response amplitudes
    mean_split_amp = np.max(split_responses.mean(axis=0), axis=0)
    mean_chat_amp = np.max(mean_chat_responses[chat_glom_ind, :, :], axis=0)

    ax3[g_ind].plot(mean_chat_amp, mean_split_amp[:30], 'kx')
    ax3[g_ind].plot([0, 0.6], [0, 0.6], 'k--')

    corr = np.corrcoef(mean_chat_amp, mean_split_amp[:30])[1, 0]
    ax3[g_ind].set_title('r = {:.2f}'.format(corr))

    # intra-individual corr for this split
    intra_ind_corr = pd.DataFrame(np.max(split_responses, axis=1).T).corr().to_numpy()[np.triu_indices(split_responses.shape[0], k=1)]
    print(np.mean(intra_ind_corr))

    for u_ind, un in enumerate(unique_parameter_values[:-2]):
        ax[g_ind, u_ind].plot(time_vector, yoffset + split_responses.mean(axis=0)[:, u_ind], color='k', alpha=1.0, linewidth=2)
        # ax[g_ind, u_ind].plot(time_vector, np.squeeze(split_responses[1, :, u_ind]).T, alpha=0.5, linewidth=2, color='k')

        ax[g_ind, u_ind].plot(time_vector, mean_chat_responses[chat_glom_ind, :, u_ind], color=colors[chat_glom_ind, :], alpha=1.0, linewidth=2)

    ax[g_ind, 0].annotate(target_glom, (0, 0.25), rotation=90)

plot_tools.addScaleBars(ax[0, 0], dT=2, dF=0.50, T_value=0, F_value=-0.08)
# fh.legend()

# %%
tt = np.max(split_responses, axis=1)
tt.shape

intra_ind_corr = pd.DataFrame(tt.T).corr()
intra_ind_corr

# %%
