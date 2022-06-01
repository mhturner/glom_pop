from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
from skimage.io import imread
from flystim import image

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')
whitened_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SaccadeSuppression',
                                                                          'include_in_analysis': True,
                                                                          'saccade_sample_period': 0.25,
                                                                          })
# %%

all_response_gains = []
all_response_amps = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)


    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix, parameter_key='current_saccade_time')
    saccade_times = [x[0] for x in unique_parameter_values]

    glom_avg_resp = np.mean(mean_response, axis=1)
    # peak response time, for each glom (across all stim conditions)
    peak_time = response_data['time_vector'][np.argmax(glom_avg_resp, axis=1)] - ID.getRunParameters('pre_time')

    resp_amp = ID.getResponseAmplitude(mean_response, metric='max')
    norm_val = resp_amp[:, -1]  # last saccade time, basically at the end of the trial, well after the response has ended
    response_gain = resp_amp / norm_val[:, np.newaxis]  # gloms x saccade times
    all_response_gains.append(response_gain)
    all_response_amps.append(resp_amp)

    # mean response trace for timing illustration
    eg_saccade_inds = np.arange(0, len(saccade_times), 1)
    eg_saccade_inds = [0, 4, 5, 6, 7, 11]
    fh0, ax0 = plt.subplots(1, 1, figsize=(4, 2))
    ax0.set_ylim([-0.1, 0.5])
    fh1, ax1 = plt.subplots(len(included_gloms), len(eg_saccade_inds), figsize=(4, 2))
    [x.set_ylim([-0.1, 0.5]) for x in ax1.ravel()]
    [x.set_axis_off() for x in ax1.ravel()]

    fh2, ax2 = plt.subplots(1, 1, figsize=(4, 2))
    ax2.set_ylim([0, 3])
    ax2.axhline(y=1, color='k', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--')

    for g_ind, glom in enumerate(included_gloms):
        for ind, si in enumerate(eg_saccade_inds):
            ax0.axvline(x=ID.getRunParameters('pre_time') + saccade_times[si], color='k', alpha=0.5)
            ax1[g_ind, ind].plot(response_data['time_vector'], mean_response[g_ind, si, :], color=util.get_color_dict()[glom])

        ax0.plot(response_data['time_vector'], glom_avg_resp[g_ind], color=util.get_color_dict()[glom])
        ax2.plot(saccade_times, response_gain[g_ind, :].T, color=util.get_color_dict()[glom], marker='o')

all_response_gains = np.stack(all_response_gains, axis=-1)
all_response_amps = np.stack(all_response_amps, axis=-1)
# %%
mean_response_gain = np.mean(all_response_gains, axis=-1)
sem_response_gain = np.std(all_response_gains, axis=-1) / np.sqrt(all_response_gains.shape[-1])

fh3, ax3 = plt.subplots(1, len(included_gloms), figsize=(8, 2))
[x.set_ylim([0.2, 1.8]) for x in ax3.ravel()]
[x.spines['top'].set_visible(False) for x in ax3.ravel()]
[x.spines['right'].set_visible(False) for x in ax3.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax3[g_ind].axhline(y=1, color='k', alpha=0.5)
    ax3[g_ind].errorbar(x=saccade_times,
                        y=mean_response_gain[g_ind, :],
                        yerr=sem_response_gain[g_ind, :],
                        color=util.get_color_dict()[glom])

    if g_ind == 0:
        ax3[g_ind].set_ylabel('Response gain')
        ax3[g_ind].set_xlabel('Time (s)')
    else:
        ax3[g_ind].set_axis_off()






# %%


# %%
