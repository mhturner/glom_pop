from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd
import ants


util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')


leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)


# %% (1) Vary rate and period


matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SurroundGratingTuning',
                                                                          'include_in_analysis': True,
                                                                          'grate_period': [5, 10, 20, 40],
                                                                          'grate_rate': [20, 40, 80, 160, 320],
                                                                          # 'spot_speed': [-100, 100]
                                                                          })

target_grate_rates = [20.,  40.,  80., 160., 320.]
target_grate_periods = [5., 10., 20., 40.]

all_responses = []
response_amps = []
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

    trial_averages = np.zeros((len(included_gloms), len(target_grate_rates), len(target_grate_periods), epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for gr_ind, gr in enumerate(target_grate_rates):
        for gp_ind, gp in enumerate(target_grate_periods):
            erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                       ID,
                                                                       query={'current_grate_rate': gr,
                                                                              'current_grate_period': gp},
                                                                       return_inds=True)

            trial_averages[:, gr_ind, gp_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x params x time
    all_responses.append(trial_averages)
    response_amps.append(ID.getResponseAmplitude(trial_averages))

    if False:  # plot individual fly responses, QC
        fh, ax = plt.subplots(len(target_grate_rates), len(target_grate_periods), figsize=(8, 4))
        fh.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax.ravel()]
        [x.set_ylim([-0.1, 0.75]) for x in ax.ravel()]
        for gr_ind, gr in enumerate(target_grate_rates):
            for gp_ind, gp in enumerate(target_grate_periods):
                for g_ind, glom in enumerate(included_gloms):
                    ax[gr_ind, gp_ind].plot(trial_averages[g_ind, gr_ind, gp_ind, :], color=util.get_color_dict()[glom])
                if gr_ind == 0:
                    ax[gr_ind, gp_ind].set_title('{:.0f}'.format(gp))
                if gp_ind == 0:
                    ax[gr_ind, gp_ind].set_ylabel('{:.0f}'.format(gr))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

# %%  Mean across animals for eg glomerulus
eg_glom = 6
print(included_gloms[eg_glom])

# fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(1.5, 2.25))
fh0, ax0 = plt.subplots(len(target_grate_periods), len(target_grate_rates), figsize=(5, 6))
[x.spines['bottom'].set_visible(False) for x in ax0.ravel()]
[x.spines['left'].set_visible(False) for x in ax0.ravel()]
[x.spines['right'].set_visible(False) for x in ax0.ravel()]
[x.spines['top'].set_visible(False) for x in ax0.ravel()]
[x.set_xticks([]) for x in ax0.ravel()]
[x.set_yticks([]) for x in ax0.ravel()]
[x.set_ylim([-0.15, 0.75]) for x in ax0.ravel()]

for gr_ind, gr in enumerate(target_grate_rates):
    for gp_ind, gp in enumerate(target_grate_periods):
        tf = gr / gp  # deg/sec / deg = 1/sec
        lbl = '{:.0f} Hz'.format(tf) if (gr_ind+gp_ind) == 0 else '{:.1f}'.format(tf).rstrip('0').rstrip('.')
        ax0[gp_ind, gr_ind].annotate(lbl, (0.5, 0.4), ha='center', color=[0.5, 0.5, 0.5], fontsize=8)

        ax0[gp_ind, gr_ind].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
        ax0[gp_ind, gr_ind].plot(response_data['time_vector'],
                                mean_responses[eg_glom, gr_ind, gp_ind, :],
                                color=util.get_color_dict()[included_gloms[eg_glom]])

        ax0[gp_ind, gr_ind].fill_between(response_data['time_vector'],
                                        mean_responses[eg_glom, gr_ind, gp_ind, :] - sem_responses[eg_glom, gr_ind, gp_ind, :],
                                        mean_responses[eg_glom, gr_ind, gp_ind, :] + sem_responses[eg_glom, gr_ind, gp_ind, :],
                                        color=util.get_color_dict()[included_gloms[eg_glom]],
                                        alpha=0.5)


        if gp_ind == 0:
            ax0[gp_ind, gr_ind].set_title('{:.0f}'.format(gr), fontsize=8)
        if gr_ind == 0:
            ax0[gp_ind, gr_ind].set_ylabel('${:.0f}\degree$'.format(gp), fontsize=8)

        if np.logical_and(gr_ind == 0, gp_ind == 0):
            plot_tools.addScaleBars(ax0[gp_ind, gr_ind], dT=2, dF=0.25, T_value=-1, F_value=-0.1)

fh0.supylabel('Spatial period ($\degree$)')
fh0.suptitle('Speed ($\degree$/s)')
# fh0.savefig(os.path.join(save_directory, 'surround_grating_{}_meantrace.svg'.format(included_gloms[eg_glom])), transparent=True)

# %% Schematic figs...
# glom map inset for highlighted glom
glom_mask_2_meanbrain = ants.image_read(os.path.join(sync_dir, 'transforms', 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')).numpy()
fh1, ax1 = plt.subplots(1, 1, figsize=(2, 1))
util.make_glom_map(ax=ax1,
                   glom_map=glom_mask_2_meanbrain,
                   z_val=None,
                   highlight_names=[included_gloms[eg_glom]])
ax1.set_axis_off()
fh1.savefig(os.path.join(save_directory, 'surround_grating_{}_glommap.svg'.format(included_gloms[eg_glom])), transparent=True)

# Sine grating image
sf = np.pi/2
xx = np.linspace(0, 2*np.pi, 200)
yy = ID.getRunParameters('grate_contrast') * np.sin(2 * np.pi * sf * xx)
img = np.repeat(yy[:, np.newaxis], 100, axis=-1)
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 0.75))
ax2.imshow(img.T, cmap='Greys_r', vmin=-1, vmax=+1)
ax2.set_axis_off()
fh2.savefig(os.path.join(save_directory, 'surround_grating_stim.svg'), transparent=True)

# TODO: Histogram of angular velocities from walking data


# %%
fh2, ax2 = plt.subplots(len(included_gloms), 2, figsize=(3.5, 6.5))
[x.set_ylim([0, 1.1]) for x in ax2.ravel()]

for g_ind, glom in enumerate(included_gloms):
    glom_data = response_amps[g_ind, :, :]
    # norm within each fly
    glom_data_norm = glom_data / np.nanmax(glom_data, axis=(0, 1))[np.newaxis, np.newaxis, :]

    period_mean = np.nanmean(glom_data_norm, axis=(0, 2))
    period_err = np.nanstd(glom_data_norm, axis=(0, 2)) / np.sqrt(glom_data.shape[-1])

    rate_mean = np.nanmean(glom_data_norm, axis=(1, 2))
    rate_err = np.nanstd(glom_data_norm, axis=(1, 2)) / np.sqrt(glom_data.shape[-1])

    ax2[g_ind, 0].errorbar(x=target_grate_rates, y=rate_mean,
                         yerr=rate_err,
                         marker='None', linestyle='-', linewidth=2, color=util.get_color_dict()[glom], alpha=0.75)

    ax2[g_ind, 1].errorbar(x=target_grate_periods, y=period_mean,
                         yerr=period_err,
                         marker='None', linestyle='-', linewidth=2, color=util.get_color_dict()[glom], alpha=0.75)


    if g_ind == 12:
        ax2[g_ind, 0].set_xticks([20, 40, 80, 160, 320])
        ax2[g_ind, 0].set_xticklabels([20, '', 80, 160, 320])
        ax2[g_ind, 0].set_yticks([0, 1.0])
        ax2[g_ind, 0].set_xlabel('Speed ($\degree$/sec)')


        ax2[g_ind, 1].set_xticks([5, 10, 20, 40])
        ax2[g_ind, 1].set_xticklabels([5, 10, 20, 40])
        ax2[g_ind, 1].set_yticks([])
        ax2[g_ind, 1].set_xlabel('Period ($\degree$)')
    else:
        ax2[g_ind, 0].set_xticks([])
        ax2[g_ind, 0].set_yticks([])

        ax2[g_ind, 1].set_xticks([])
        ax2[g_ind, 1].set_yticks([])


    ax2[g_ind, 0].spines['top'].set_visible(False)
    ax2[g_ind, 0].spines['right'].set_visible(False)
    ax2[g_ind, 1].spines['top'].set_visible(False)
    ax2[g_ind, 1].spines['right'].set_visible(False)

fh2.supylabel('Response amplitude (normalized)')
fh2.savefig(os.path.join(save_directory, 'surround_grating_tuning.svg'), transparent=True)


# %% (2) VARY DIRECTION
target_angles = [0, 45, 90, 135, 180, 225, 270, 315]
eg_series = ('2022-05-26', 3)  # ('2022-05-26', 3)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SurroundGratingTuning',
                                                                          'include_in_analysis': True,
                                                                          'angle': target_angles
                                                                          })



all_responses = []
response_amps = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)
    behaving_trials = np.where(behavior_data.get('behaving'))[0]
    nonbehaving_trials = np.where(~behavior_data.get('behaving'))[0]

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    trial_averages = np.zeros((len(included_gloms), len(target_angles), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for ang_ind, ang in enumerate(target_angles):
        erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                   ID,
                                                                   query={'current_angle': ang},
                                                                   return_inds=True)

        behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, ang_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, ang_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time

        # trial_averages[:, ang_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x params x time
    all_responses.append(trial_averages)
    response_amps.append(ID.getResponseAmplitude(trial_averages))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    # if True: # QC
        fh0, ax0 = plt.subplots(len(included_gloms), len(target_angles), figsize=(2, 3))
        # fh.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax0.ravel()]
        [x.set_ylim([-0.1, 0.4]) for x in ax0.ravel()]
        for ang_ind, ang in enumerate(target_angles):
            for g_ind, glom in enumerate(included_gloms):
                ax[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 0,  :], color='b')
                ax[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 1, :], color='k')
            if g_ind == 0:
                ax0[gr_ind, gp_ind].set_title('{:.0f}'.format(ang))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

# %%  Polar plots, mean + individual flies
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
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)


# %% (1) SPEED & IMAGE & FILTER

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          'image_speed': [0, 40, 160, 320],
                                                                          'filter_flag': [0, 1, 3, 4],
                                                                          'image_index': [0, 5, 15],
                                                                          })


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


new_im = image.Image(image_name=image_names[0])
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


# %%

fh, ax = plt.subplots(1, 4, figsize=(8, 2))
ax[0].hist(img_orig.ravel(), bins=100);
ax[1].hist(img_white.ravel(), bins=100)
ax[2].hist(img_hp.ravel(), bins=100)
ax[3].hist(img_lp.ravel(), bins=100);


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
            pull_speed_ind = np.where([320 == x[2] for x in unique_parameter_values])[0]

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


# %% (2) 8 DIRECTIONS
target_angles = [0, 45, 90, 135, 180, 225, 270, 315]
eg_series = ('2022-05-24', 5)  # ('2022-05-24', 5)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'SurroundGratingTuning',
                                                                          'include_in_analysis': True,
                                                                          'grate_rate': [160],
                                                                          'angle': target_angles,
                                                                          })

all_responses = []
response_amps = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)


    # Get behavior data
    behavior_data = dataio.load_behavior(ID, process_behavior=True)
    behaving_trials = np.where(behavior_data.get('behaving'))[0]
    nonbehaving_trials = np.where(~behavior_data.get('behaving'))[0]

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    trial_averages = np.zeros((len(included_gloms), len(target_angles), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    for ang_ind, ang in enumerate(target_angles):
        erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                   ID,
                                                                   query={'current_angle': ang},
                                                                   return_inds=True)

        behaving_inds = np.array([x for x in matching_inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, ang_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in matching_inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, ang_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time

        # trial_averages[:, ang_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x params x time
    all_responses.append(trial_averages)
    response_amps.append(ID.getResponseAmplitude(trial_averages))

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    # if True:  # plot individual fly responses, QC
        fh0, ax0 = plt.subplots(len(included_gloms), len(target_angles), figsize=(3, 4))
        # fh.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax0.ravel()]
        [x.set_ylim([-0.1, 0.4]) for x in ax0.ravel()]
        for ang_ind, ang in enumerate(target_angles):
            ax0[0, ang_ind].set_title('{}$\degree$'.format(ang), rotation=45)
            for g_ind, glom in enumerate(included_gloms):
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 1, :], color='k', label='Nonbehaving' if (g_ind+ang_ind == 0) else None)
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 0,  :], color='b', label='Nonbehaving' if (g_ind+ang_ind == 0) else None)
            if g_ind == 0:
                ax0[gr_ind, gp_ind].set_title('{:.0f}'.format(ang))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)

fh0.legend()
# %%


fh1, ax1 = plt.subplots(6, 1, figsize=(3, 6), subplot_kw={'projection': 'polar'})
ax1 = ax1.ravel()

mean_tuning = []
for f_ind in range(len(matching_series)):
    fly_tuning = []
    for b in range(2):
        dir_tuning = ID.getResponseAmplitude(all_responses[:, :, b, :, f_ind])
        fly_tuning.append(dir_tuning)

        for g_ind, glom in enumerate(included_gloms):
            dir_resp = dir_tuning[g_ind, :]
            plot_resp = np.append(dir_resp, dir_resp[0])
            plot_dir = np.append(target_angles, target_angles[0])
            if b == 0:
                color = 'b'
            elif b == 1:
                color = 'k'
            # ax1[g_ind].plot(np.deg2rad(plot_dir), plot_resp, color=color, alpha=0.5)

    mean_tuning.append(np.stack(fly_tuning, axis=-1))

mean_tuning = np.stack(mean_tuning, axis=-1)
mean_tuning = np.nanmean(mean_tuning, axis=-1) # glom x dir x beh/nonbeh

# Plot mean dir tuning across flies, for beh vs. nonbeh trials
for g_ind, glom in enumerate(included_gloms):
    plot_dir = np.append(target_angles, target_angles[0])
    # Behaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 0], mean_tuning[g_ind, 0, 0])
    ax1[g_ind].plot(np.deg2rad(plot_dir), plot_resp,
                    color='b', linewidth=2, marker='.',
                    label='Behaving' if (g_ind == 0) else None)

    # Nonbehaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 1], mean_tuning[g_ind, 0, 1])
    ax1[g_ind].plot(np.deg2rad(plot_dir), plot_resp,
                    color='k', linewidth=2, marker='.',
                    label='Nonbehaving' if (g_ind == 0) else None)


    ax1[g_ind].annotate(glom, (0, 0), ha='center')
fh1.legend()
# %%
