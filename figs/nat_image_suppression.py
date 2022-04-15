from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
from skimage.io import imread

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')


leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          # 'image_speed': [-40.,   0.,  40.,  80., 120.],
                                                                          'image_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                          })

# %%
plot_glom_ind = 0
all_responses = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Align responses
    parameter_key = ['stim0_image_name', 'current_filter_flag', 'current_image_speed']
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix, parameter_key=parameter_key)

    all_responses.append(mean_response)

# %%


def get_vh_image(image_name):
    return imread(os.path.join(images_dir, image_name))


image_names = np.unique([x[0].replace('whitened_', '') for x in unique_parameter_values])
filter_codes = np.unique([x[1] for x in unique_parameter_values])
image_speeds = np.unique([x[2] for x in unique_parameter_values])

# Shape = gloms, images, speeds, filter, flies
response_amps = np.zeros((len(included_gloms), len(image_names), len(image_speeds), 2, len(all_responses)))
for fly_ind, mean_response in enumerate(all_responses):
    fly_resp_amps = ID.getResponseAmplitude(mean_response, metric='max')
    fh, ax = plt.subplots(len(image_names), len(image_speeds)+1, gridspec_kw={'width_ratios': [1, 1, 4]})
    [plot_tools.cleanAxes(x) for x in ax.ravel()]
    [x.set_ylim([-0.1, 0.75]) for x in ax[:, :-1].ravel()]
    for im_ind, image_name in enumerate(image_names):
        ax[im_ind, -1].imshow(get_vh_image(image_name), cmap='Greys_r')
        for spd_ind, image_speed in enumerate(image_speeds):
            for fc_ind, filter_code in enumerate(filter_codes):
                pull_image_ind = np.where([image_name in x[0] for x in unique_parameter_values])[0]
                pull_filter_ind = np.where([filter_code == x[1] for x in unique_parameter_values])[0]
                pull_speed_ind = np.where([image_speed == x[2] for x in unique_parameter_values])[0]

                pull_ind = list(set.intersection(set(pull_image_ind),
                                                 set(pull_filter_ind),
                                                 set(pull_speed_ind)))


                assert len(pull_ind) == 1
                pull_ind = pull_ind[0]

                response_amps[:, im_ind, spd_ind, fc_ind, fly_ind] = fly_resp_amps[:, pull_ind]

                if filter_code == 0:
                    alpha = 1.0
                else:
                    alpha = 0.5

                ax[im_ind, spd_ind].plot(mean_response[plot_glom_ind, pull_ind, :], 'k', alpha=alpha)
                if im_ind == 0:
                    ax[im_ind, spd_ind].set_title('{:.0f}'.format(image_speed),
                                                  color='r' if image_speed==ID.getRunParameters('spot_speed') else 'k')

# %%

# static vs moving background image
fh, ax = plt.subplots(len(included_gloms), 1, figsize=(1.5, 8))
ax = ax.ravel()
[x.set_xlim([0, 1]) for x in ax]
[x.set_ylim([0, 1]) for x in ax]
[x.plot([0, 1], [0, 1], 'k-', alpha=0.5) for x in ax]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].set_title(glom, rotation=0)
    # Mean +/- sem across flies, for each image
    across_fly_mean_static = response_amps[g_ind, :, 0, 0, :].mean(axis=-1)
    across_fly_mean_moving = response_amps[g_ind, :, 1, 0, :].mean(axis=-1)
    ax[g_ind].plot(across_fly_mean_static,
                   across_fly_mean_moving, 'o', color=util.get_color_dict()[glom])

fh.supxlabel('Static background')
fh.supylabel('Moving background')

# Whitened vs original image
fh, ax = plt.subplots(len(included_gloms), 1, figsize=(1.5, 8))
ax = ax.ravel()
[x.set_xlim([0, 1]) for x in ax]
[x.set_ylim([0, 1]) for x in ax]
[x.plot([0, 1], [0, 1], 'k-', alpha=0.5) for x in ax]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].set_title(glom, rotation=0)
    # Mean +/- sem across flies, for each image
    across_fly_mean_static = response_amps[g_ind, :, 1, 0, :].mean(axis=-1)
    across_fly_mean_moving = response_amps[g_ind, :, 1, 1, :].mean(axis=-1)
    ax[g_ind].plot(across_fly_mean_static,
                   across_fly_mean_moving, 'o', color=util.get_color_dict()[glom])

fh.supxlabel('Original image')
fh.supylabel('Whitened image')

        # %%

all_amplitudes = ID.getResponseAmplitude(all_responses[0], metric='max')

fh, ax = plt.subplots(len(included_gloms), 1, figsize=(0.75, 8))
[x.set_xticks([]) for x in ax]
[x.set_yticks([]) for x in ax]
[x.set_xlim([0, 0.7]) for x in ax]
[x.set_ylim([0, 0.7]) for x in ax]
[x.spines['top'].set_visible(False) for x in ax]
[x.spines['right'].set_visible(False) for x in ax]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].plot([0, 0.7], [0, 0.7], 'k-', alpha=0.5, zorder=0)
    ax[g_ind].scatter(all_amplitudes[g_ind, :, 0, :].mean(axis=0),
                      all_amplitudes[g_ind, :, 1, :].mean(axis=0),
                      color=util.get_color_dict()[glom])
    result = ttest_rel(all_amplitudes[g_ind, :, 0, :].mean(axis=0),
                       all_amplitudes[g_ind, :, 1, :].mean(axis=0))





# %%
