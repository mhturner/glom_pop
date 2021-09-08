from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc

from glom_pop import dataio, util


experiment_file_directory = '/Users/mhturner/CurrentData'

# brain_file = ('2021-08-04', 2)
# brain_file = ('2021-08-04', 5)
# brain_file = ('2021-08-04', 8)
#
# brain_file = ('2021-08-11', 2)
# brain_file = ('2021-08-11', 5)
# brain_file = ('2021-08-11', 8)
#
# brain_file = ('2021-08-20', 3)
# brain_file = ('2021-08-20', 4) # spot on vr
# brain_file = ('2021-08-20', 8)
#
# brain_file = ('2021-08-25', 3)
brain_file = ('2021-08-25', 11)


experiment_file_name = brain_file[0]
series_number = brain_file[1]

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = volumetric_data.VolumetricDataObject(file_path,
                                          series_number,
                                          quiet=False)

# Load response data
response_data = dataio.loadResponses(ID, response_set_name='glom')

vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

meanbrain_red = response_data.get('meanbrain')[..., 0]
meanbrain_green = response_data.get('meanbrain')[..., 1]

# Align responses
mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, trial_response_by_stimulus = ID.getMeanBrainByStimulus(response_data.get('epoch_response'), parameter_key='current_trajectory_index')
n_stimuli = mean_voxel_response.shape[2]
concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1) # responses, time (concat stims)


cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())

# %%
ID.getStimulusTiming(plot_trace_flag=True)
# %%
len(trial_response_by_stimulus)
trial_response_by_stimulus[1].shape


# %% STABILITY ACROSS TRIALS

glom_ind = 1
fh, ax = plt.subplots(n_stimuli, 1, figsize=(8, 6))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 0.75]) for x in ax.ravel()]

for s_ind in range(n_stimuli):
    trials = trial_response_by_stimulus[s_ind].shape[2]

    for t in range(trials):
        ax[s_ind].plot(response_data.get('time_vector'), trial_response_by_stimulus[s_ind][glom_ind, :, t], alpha=0.5, color='k')

    ax[s_ind].plot(response_data.get('time_vector'), trial_response_by_stimulus[s_ind][glom_ind, :, :].mean(axis=-1), color='b', alpha=1.0, linewidth=3)
    if (s_ind == 0):
        plot_tools.addScaleBars(ax[s_ind], dT=1, dF=0.50, T_value=-0.1, F_value=-0.2)


# %%

fh, ax = plt.subplots(14, 1, figsize=(16, 8))
for i in range(14):
    ax[i].plot(response_data['response'][i, :])
    ax[i].set_axis_off()
# %% Glom responses
fh, ax = plt.subplots(1 + concatenated_tuning.shape[0], len(unique_parameter_values), figsize=(18, 18))
[util.cleanAxes(x) for x in ax.ravel()]
[x.set_ylim([-0.25, 0.75]) for x in ax.ravel()]

fh.subplots_adjust(wspace=0.05, hspace=0.05)

for u_ind, un in enumerate(unique_parameter_values):
    for g_ind, name in enumerate(names):
        ax[g_ind+1, u_ind].plot(response_data.get('time_vector'), mean_voxel_response[g_ind, :, u_ind], color=colors[g_ind, :])
        ax[g_ind+1, u_ind].axhline(color='k', alpha=0.5)
        if (g_ind == 0) & (u_ind == 0):
            plot_tools.addScaleBars(ax[g_ind+1, u_ind], dT=1, dF=0.25, T_value=0, F_value=-0.2)

        if u_ind == 0:
            ax[g_ind+1, u_ind].set_ylabel(name)
