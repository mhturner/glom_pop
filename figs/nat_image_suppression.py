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
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          'image_speed': [0, 40, 160, 320],
                                                                          # 'image_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                          'image_index': [0, 5, 15],
                                                                          })


# %% Pull out amp stats for each image/speed/filter/fly

all_responses = []
response_amps = []
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

    # get response amps and sort by image, speed, filter code
    tmp_ra = ID.getResponseAmplitude(mean_response, metric='max')

    image_names = np.unique([x[0].replace('whitened_', '') for x in unique_parameter_values])
    filter_codes = np.unique([x[1] for x in unique_parameter_values])
    image_speeds = np.unique([x[2] for x in unique_parameter_values])
    new_resp_amp = np.zeros((len(included_gloms), len(image_names), len(image_speeds), len(filter_codes)))
    for im_ind, image_name in enumerate(image_names):
        for spd_ind, image_speed in enumerate(image_speeds):
            for fc_ind, filter_code in enumerate(filter_codes):
                pull_image_ind = np.where([image_name in x[0] for x in unique_parameter_values])[0]
                pull_filter_ind = np.where([filter_code == x[1] for x in unique_parameter_values])[0]
                pull_speed_ind = np.where([image_speed == x[2] for x in unique_parameter_values])[0]

                pull_ind = list(set.intersection(set(pull_image_ind),
                                                 set(pull_filter_ind),
                                                 set(pull_speed_ind)))

                new_resp_amp[:, im_ind, spd_ind, fc_ind] = tmp_ra[:, pull_ind[0]]

    response_amps.append(new_resp_amp)

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

# %% Get filtered images for fig. panels
pixels_per_degree = 1536 / 360
screen_width = 160 * pixels_per_degree  # deg -> pixels
screen_height = 50 * pixels_per_degree  # deg -> pixels


new_im = image.Image(image_name=image_names[1])
img_orig = new_im.load_image()
image_width = img_orig.shape[1]
image_height = img_orig.shape[0]
freq, pspect_orig = util.get_power_spectral_density(img_orig[:, 512:2*512], pixels_per_degree)

# High-pass
filter_name = 'butterworth'
filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                 'order': 2,
                 'high_pass': True}

img_hp = new_im.filter_image(filter_name=filter_name,
                             filter_kwargs=filter_kwargs)
freq, pspect_hp = util.get_power_spectral_density(img_hp[:, 512:2*512], pixels_per_degree)

# Low-pass
filter_name = 'butterworth'
filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                 'order': 2,
                 'high_pass': False}

img_lp = new_im.filter_image(filter_name=filter_name,
                             filter_kwargs=filter_kwargs)
freq, pspect_lp = util.get_power_spectral_density(img_lp[:, 512:2*512], pixels_per_degree)

img_white = new_im.whiten_image()

# %% mean resp for eg glom
eg_glom = 0

filter_list = ['Original', 'Whitened', 'DoG', 'Highpass', 'Lowpass']
colors = 'kbxmy'

# fh0: original image only, no filtering
fh0, ax0 = plt.subplots(len(image_names), len(image_speeds)+1, figsize= (4, 2), gridspec_kw={'width_ratios': [1.5, 1, 1, 1, 1]})
[plot_tools.cleanAxes(x) for x in ax0.ravel()]
[x.set_ylim([-0.15, 0.6]) for x in ax0[:, 1:].ravel()]

# # fh2: for spd=160, compare filter conditions
fh2, ax2 = plt.subplots(len(image_names), len(filter_codes), figsize= (3, 2))
[plot_tools.cleanAxes(x) for x in ax2.ravel()]
[x.set_ylim([-0.15, 0.6]) for x in ax2.ravel()]




for im_ind, image_name in enumerate(image_names):
    ax0[im_ind, 0].imshow(np.flipud(image.Image(image_name).load_image()), cmap='Greys_r')
    ax0[im_ind, 0].set_axis_on()
    ax0[im_ind, 0].set_xticks([])
    ax0[im_ind, 0].set_yticks([])

    ax0[im_ind, 0].set_ylabel('Im. {}'.format(im_ind+1))
    for spd_ind, image_speed in enumerate(image_speeds):
        if im_ind == 0:
            ax0[im_ind, spd_ind+1].set_title('{:.0f}$\degree$/s'.format(image_speed))
        if np.logical_and(im_ind == 0, spd_ind == 0):
            plot_tools.addScaleBars(ax0[im_ind, spd_ind+1], dT=2, dF=0.25, T_value=-1, F_value=-0.1)
        pull_image_ind = np.where([image_name in x[0] for x in unique_parameter_values])[0]
        pull_filter_ind = np.where([0 == x[1] for x in unique_parameter_values])[0]
        pull_speed_ind = np.where([image_speed == x[2] for x in unique_parameter_values])[0]

        pull_ind = list(set.intersection(set(pull_image_ind),
                                         set(pull_filter_ind),
                                         set(pull_speed_ind)))

        ax0[im_ind, spd_ind+1].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
        ax0[im_ind, spd_ind+1].plot(response_data['time_vector'], mean_responses[eg_glom, pull_ind, :][0, :],
                                    color=util.get_color_dict()[included_gloms[eg_glom]])
        ax0[im_ind, spd_ind+1].fill_between(response_data['time_vector'],
                                            mean_responses[eg_glom, pull_ind, :][0, :] - sem_responses[eg_glom, pull_ind, :][0, :] ,
                                            mean_responses[eg_glom, pull_ind, :][0, :] + sem_responses[eg_glom, pull_ind, :][0, :] ,
                                            color=util.get_color_dict()[included_gloms[eg_glom]], alpha=0.25, linewidth=0)
        for fc_ind, filter_code in enumerate(filter_codes):
            if im_ind == 0:
                ax2[im_ind, fc_ind].set_title(filter_list[int(filter_code)], fontsize=10, rotation=-45)

            if np.logical_and(im_ind == 0, fc_ind == 0):
                plot_tools.addScaleBars(ax2[im_ind, fc_ind], dT=2, dF=0.25, T_value=-1, F_value=-0.1)
            pull_image_ind = np.where([image_name in x[0] for x in unique_parameter_values])[0]
            pull_filter_ind = np.where([filter_code == x[1] for x in unique_parameter_values])[0]
            pull_speed_ind = np.where([160 == x[2] for x in unique_parameter_values])[0]

            pull_ind = list(set.intersection(set(pull_image_ind),
                                             set(pull_filter_ind),
                                             set(pull_speed_ind)))



            ax2[im_ind, fc_ind].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax2[im_ind, fc_ind].fill_between(response_data['time_vector'],
                                                mean_responses[eg_glom, pull_ind, :][0, :] - sem_responses[eg_glom, pull_ind, :][0, :] ,
                                                mean_responses[eg_glom, pull_ind, :][0, :] + sem_responses[eg_glom, pull_ind, :][0, :] ,
                                                color=util.get_color_dict()[included_gloms[eg_glom]], alpha=0.25, linewidth=0)
            ax2[im_ind, fc_ind].plot(response_data['time_vector'], mean_responses[eg_glom, pull_ind, :].T,
                                       color=util.get_color_dict()[included_gloms[eg_glom]], linewidth=1,
                                       label=filter_list[int(filter_code)] if im_ind+spd_ind == 0 else '')



fh0.savefig(os.path.join(save_directory, 'nat_image_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)
fh2.savefig(os.path.join(save_directory, 'nat_filter_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)

# %% pop stats: response as a fxn of speed for filter conditions

fh1, ax1 = plt.subplots(len(included_gloms), 3, figsize=(3, 5.5))
[x.set_ylim([0, 1.1]) for x in ax1.ravel()]
[x.spines['top'].set_visible(False) for x in ax1.ravel()]
[x.spines['right'].set_visible(False) for x in ax1.ravel()]

# for im_ind, image_name in enumerate(image_names):

for fc_ind, fc in enumerate(filter_codes):
    for g_ind, glom in enumerate(included_gloms):

        if fc_ind == 0: # original image, plot to all panels

            fly_responses = response_amps[g_ind, :, :, 0, :]  # image x speed x flies
            fly_responses_norm = fly_responses / np.nanmax(fly_responses, axis=(0, 1))[np.newaxis, np.newaxis, :]
            meanresp = np.nanmean(fly_responses_norm, axis=(0, 2)) # average over flies and images
            semresp = np.nanstd(fly_responses_norm, axis=(0, 2)) / np.sqrt(fly_responses.shape[-1])
            for panel in range(3):
                ax1[g_ind, panel].errorbar(x=image_speeds, y=meanresp,
                                            yerr=semresp,
                                            marker='None', linestyle='-', linewidth=2, color=util.get_color_dict()[glom], alpha=0.75)


        else:  # filter conditions
            if g_ind == 0:
                ax1[g_ind, fc_ind-1].set_title(filter_list[int(fc)])
            fly_responses = response_amps[g_ind, :, :, fc_ind, :]  # image x speed x flies
            fly_responses_norm = fly_responses / np.nanmax(fly_responses, axis=(0, 1))[np.newaxis, np.newaxis, :]
            meanresp = np.nanmean(fly_responses_norm, axis=(0, 2)) # average over flies and images
            semresp = np.nanstd(fly_responses_norm, axis=(0, 2)) / np.sqrt(fly_responses.shape[-1])
            ax1[g_ind, fc_ind-1].errorbar(x=image_speeds, y=meanresp,
                                        yerr=semresp,
                                        marker='None', linestyle=':', linewidth=2, color='k', alpha=0.75)


[x.set_xticks([]) for x in ax1.ravel()]
[x.set_yticks([]) for x in ax1.ravel()]

ax1[12, 0].set_xticks([0, 160, 320])
ax1[12, 0].set_yticks([0, 1])

ax1[12, 1].set_xlabel('Speed ($\degree$/s)')
fh1.supylabel('Response amplitude (normalized)')
fh1.savefig(os.path.join(save_directory, 'nat_image_popstats.svg'), transparent=True)


# %%
