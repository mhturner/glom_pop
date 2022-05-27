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
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SaccadeSuppression',
                                                                          'include_in_analysis': True,

                                                                          })

# %%
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

# %%
window_size_sec = 4.0  # sec
window_size_frames = int(np.ceil(window_size_sec / ID.getAcquisitionMetadata('sample_period')))  # sec -> imaging frames


series = matching_series[0]
series_number = series['series']
file_path = series['file_name'] + '.hdf5'
file_name = os.path.split(series['file_name'])[-1]
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# Load response data
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)


fh, ax = plt.subplots(30, 1, figsize=(8, 20))
[x.set_axis_off() for x in ax.ravel()]
[x.set_ylim([-0.1, 0.75]) for x in ax.ravel()]
snippets = []
for e in range(30):
    saccade_timepoints = ID.getRunParameters('pre_time') + ID.getEpochParameters('saccade_timepoints')[e]
    saccade_theta = ID.getEpochParameters('saccade_theta')[e]
    ax[e].plot(response_data.get('time_vector'), epoch_response_matrix[0, e, :], 'k')

    for st in saccade_timepoints[1:-1]:
        ax[e].plot([st, st], [0, 1], color=[0.5, 0.5, 0.5], linestyle='-', alpha=0.5)

    saccade_starts = saccade_timepoints[1:-1][0::2]
    for ss in saccade_starts:
        pull_start = np.where(response_data.get('time_vector') > (ss-window_size_sec/2))[0][0]
        pull_stop = pull_start + window_size_frames
        snippets.append(epoch_response_matrix[0, e, pull_start:pull_stop])


snippets = np.vstack(snippets)

# %%
fh, ax = plt.subplots(1, 1, figsize=(4, 4))
# ax.plot(snippets.T, alpha=0.5, color=[0.5, 0.5, 0.5])
ax.plot(snippets.mean(axis=0), 'k-o')
ax.axvline(x=7, color='b')


# %%
saccade_timepoints[1:-1]
saccade_timepoints[1:-1][0::2]
saccade_timepoints

trial_avg = epoch_response_matrix.mean(axis=1)
trial_avg.shape

fh, ax = plt.subplots(13, 1, figsize=(5, 8))
[x.set_axis_off() for x in ax.ravel()]
[x.set_ylim([-0.1, 0.4]) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].plot(trial_avg[g_ind], color=util.get_color_dict()[glom])

# %%
