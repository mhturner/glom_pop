from visanalysis.analysis import imaging_data
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util

util.config_matplotlib()

base_dir = dataio.get_config_file()['base_dir']
experiment_file_directory = dataio.get_config_file()['experiment_file_directory']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(base_dir, 'transforms', 'meanbrain_template')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

glom_size_threshold = 10

all_responses = []
response_amplitudes = []
all_glom_sizes = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    glom_sizes = np.zeros(len(included_vals))
    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)
        glom_sizes[val_ind] = new_glom_size

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    # Align responses
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix, parameter_key='current_coherence')

    response_amp = ID.getResponseAmplitude(mean_response, metric='max')

    all_responses.append(mean_response)
    response_amplitudes.append(response_amp)
    all_glom_sizes.append(glom_sizes)


# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)
all_glom_sizes = np.stack(all_glom_sizes, axis=-1)  # dims = (gloms, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

# %% Plot resp. vs. dot coherence

fh, ax = plt.subplots(len(included_gloms), len(unique_parameter_values), figsize=(6, 7))
[x.set_ylim([-0.15, 0.3]) for x in ax.ravel()]
# [x.set_axis_off() for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind, 0].set_ylabel(glom)
    for u_ind, up in enumerate(unique_parameter_values):
        ax[0, u_ind].set_title(up)
        ax[g_ind, u_ind].plot(mean_responses[g_ind, u_ind, :], color=util.get_color_dict()[glom])
        # ax[g_ind, u_ind].plot(all_responses[g_ind, u_ind, :, :], alpha=0.5, color='k')

clusters = [['LPLC2', 'LPLC1', 'LC6', 'LC26', 'LC16', 'LC4'],
            ['LC18', 'LC9', 'LC15', 'LC12', 'LC17'],
            ['LC11', 'LC21']]


fh, ax = plt.subplots(3, 1, figsize=(2.0, 4.5))
[x.set_ylim([0, 0.3]) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):

    cluster_ind = np.where([glom in c for c in clusters])[0]
    if len(cluster_ind) > 0:
        ax[cluster_ind[0]].plot(unique_parameter_values, response_amplitudes[g_ind, :, :].mean(axis=-1),
                                color=util.get_color_dict()[glom], marker='o', linestyle='-')

ax[2].set_xlabel('Coherence')
ax[1].set_ylabel('Mean response (dF/F)')
cluster_ind

# %%