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
                                                                          'image_speed': [0, 40, 160, 320],
                                                                          # 'image_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                          'image_index': [0, 5, 15],
                                                                          })

# %%
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

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)
# %%

def get_vh_image(image_name):
    return imread(os.path.join(images_dir, image_name))

image_names = np.unique([x[0].replace('whitened_', '') for x in unique_parameter_values])
filter_codes = np.unique([x[1] for x in unique_parameter_values])
image_speeds = np.unique([x[2] for x in unique_parameter_values])

eg_glom = 0

fh0, ax0 = plt.subplots(len(image_names), len(image_speeds)+1, figsize= (5, 3), gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})
[plot_tools.cleanAxes(x) for x in ax0.ravel()]
[x.set_ylim([-0.15, 0.6]) for x in ax0[:, 1:].ravel()]
colors = 'kbxmy'
filter_list = ['Original', 'Whitened', 'DoG', 'Highpass', 'Lowpass']
ax0[0, 0].set_title('Background image')
for im_ind, image_name in enumerate(image_names):
    ax0[im_ind, 0].imshow(np.flipud(get_vh_image(image_name)), cmap='Greys_r')
    for spd_ind, image_speed in enumerate(image_speeds):
        if im_ind == 0:
            ax0[im_ind, spd_ind+1].set_title('{:.0f}$\degree$/s'.format(image_speed))
        if np.logical_and(im_ind == 0, spd_ind == 0):
            plot_tools.addScaleBars(ax0[im_ind, spd_ind+1], dT=2, dF=0.25, T_value=-1, F_value=-0.1)
        for fc_ind, filter_code in enumerate(filter_codes):
            pull_image_ind = np.where([image_name in x[0] for x in unique_parameter_values])[0]
            pull_filter_ind = np.where([filter_code == x[1] for x in unique_parameter_values])[0]
            pull_speed_ind = np.where([image_speed == x[2] for x in unique_parameter_values])[0]

            pull_ind = list(set.intersection(set(pull_image_ind),
                                             set(pull_filter_ind),
                                             set(pull_speed_ind)))

            ax0[im_ind, spd_ind+1].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax0[im_ind, spd_ind+1].plot(response_data['time_vector'], mean_responses[eg_glom, pull_ind, :].T,
                                        color=colors[int(filter_code)],
                                        label=filter_list[int(filter_code)] if im_ind+spd_ind == 0 else '')

fh0.suptitle('                                                     Image speed')
fh0.legend()
fh0.savefig(os.path.join(save_directory, 'nat_image_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)



# %% OLD OLD OLD
# %% TODO: summary plots for nat image suppression
# response_amps shape = (gloms, images, speeds, filters flies)
#mod_index shape = (gloms, images, filters flies)
mod_index = response_amps[:, :, 1, :, :] / response_amps[:, :, 0, :, :]

fh, ax = plt.subplots(len(included_gloms), 1, figsize=(6, 8))
for g_ind, glom in enumerate(included_gloms):

    ct, bn = np.histogram(mod_index[g_ind, :, 0, :].ravel(),
                          # bins=np.linspace(0, np.nanmax(mod_index), 40),
                          bins=30,
                          density=False)
    bn_ctr = bn[:-1] + np.diff(bn)[0]
    bn_prob = ct / np.sum(ct)
    ax[g_ind].fill_between(bn_ctr, bn_prob, color=util.get_color_dict()[glom])

    ct, bn = np.histogram(mod_index[g_ind, :, 1, :].ravel(),
                          # bins=np.linspace(0, np.nanmax(mod_index), 40),
                          bins=30,
                          density=False)
    bn_ctr = bn[:-1] + np.diff(bn)[0]
    bn_prob = ct / np.sum(ct)
    ax[g_ind].fill_between(bn_ctr, bn_prob, color=util.get_color_dict()[glom], alpha=0.5)

    ax[g_ind].axvline(x=1)

# %%

# static vs moving background image
fh, ax = plt.subplots(4, 4, figsize=(4, 4))
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
fh, ax = plt.subplots(4, 4, figsize=(4, 4))
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
