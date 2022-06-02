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
import pandas as pd
import seaborn as sns

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


# %% (1) SPEED & IMAGE & FILTER
image_speeds = [0, 40, 160, 320]
filter_codes = [0, 1, 3, 4]
matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          'image_speed': image_speeds,
                                                                          'filter_flag': filter_codes,
                                                                          'image_index': [0, 5, 15],
                                                                          })



# %%
all_resp_mat = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('-----')
    print('File = {} / series = {}'.format(file_name, series_number))
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)



    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
    resp_mat = np.zeros((len(included_gloms), len(image_speeds), len(filter_codes), epoch_response_matrix.shape[-1]))

    for spd_ind, image_speed in enumerate(image_speeds):
        for fc_ind, filter_code in enumerate(filter_codes):
            erm_selected, matching_inds = shared_analysis.filterTrials(epoch_response_matrix,
                                                                       ID,
                                                                       query={'current_filter_flag': filter_code,
                                                                              'current_image_speed': image_speed},
                                                                       return_inds=True)
            resp_mat[:, spd_ind, fc_ind, :] =  np.nanmean(erm_selected, axis=1)

    all_resp_mat.append(resp_mat)

    print('-----')
all_resp_mat = np.stack(all_resp_mat, axis=-1)  # shape = (gloms, speeds, filters, flies)

# %% Mean traces across animals for eg glomerulus
eg_glom_ind = 0

filter_list = ['Original', 'Whitened', 'DoG', 'Highpass', 'Lowpass']
fh0, ax0 = plt.subplots(len(filter_codes), len(image_speeds), figsize=(2.5, 3.5))
[x.set_ylim([-0.15, 0.6]) for x in ax0.ravel()]
[x.spines['bottom'].set_visible(False) for x in ax0.ravel()]
[x.spines['left'].set_visible(False) for x in ax0.ravel()]
[x.spines['right'].set_visible(False) for x in ax0.ravel()]
[x.spines['top'].set_visible(False) for x in ax0.ravel()]
[x.set_xticks([]) for x in ax0.ravel()]
[x.set_yticks([]) for x in ax0.ravel()]
for spd_ind, image_speed in enumerate(image_speeds):
    for fc_ind, filter_code in enumerate(filter_codes):
        mean_resp = np.mean(all_resp_mat[eg_glom_ind, spd_ind, fc_ind, :], axis=-1)
        sem_resp = np.std(all_resp_mat[eg_glom_ind, spd_ind, fc_ind, :], axis=-1) / all_resp_mat.shape[-1]
        ax0[fc_ind, spd_ind].axhline(y=0, color='k', alpha=0.5)
        ax0[fc_ind, spd_ind].fill_between(response_data['time_vector'],
                                          mean_resp - sem_resp,
                                          mean_resp + sem_resp,
                                          color=util.get_color_dict()[included_gloms[eg_glom_ind]],
                                          alpha=0.5)
        ax0[fc_ind, spd_ind].plot(response_data['time_vector'], mean_resp,
                                  color=util.get_color_dict()[included_gloms[eg_glom_ind]])

        if fc_ind == 0:
            ax0[fc_ind, spd_ind].set_title('{}'.format(image_speed))
            if spd_ind == 0:
                plot_tools.addScaleBars(ax0[fc_ind, spd_ind], dT=2, dF=0.25, T_value=-0.1, F_value=-0.08)
        if spd_ind == 0:
            ax0[fc_ind, spd_ind].set_ylabel(filter_list[filter_code], rotation=45)

fh0.suptitle('Background speed ($\degree$/s)')
fh0.savefig(os.path.join(save_directory, 'natimage_traces_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)

# %%
# eg single fly glom to original image only
eg_fly_ind = 1
fh1, ax1 = plt.subplots(1, 4, figsize=(2.5, 0.7))
[x.set_ylim([-0.15, 0.8]) for x in ax1.ravel()]
[x.spines['bottom'].set_visible(False) for x in ax1.ravel()]
[x.spines['left'].set_visible(False) for x in ax1.ravel()]
[x.spines['right'].set_visible(False) for x in ax1.ravel()]
[x.spines['top'].set_visible(False) for x in ax1.ravel()]
[x.set_xticks([]) for x in ax1.ravel()]
[x.set_yticks([]) for x in ax1.ravel()]
for spd_ind, image_speed in enumerate(image_speeds):
    ax1[spd_ind].axhline(y=0, color='k', alpha=0.5)
    ax1[spd_ind].plot(response_data['time_vector'],
                      all_resp_mat[eg_glom_ind, spd_ind, 0, :, eg_fly_ind],
                      color=util.get_color_dict()[included_gloms[eg_glom_ind]])
    ax1[spd_ind].set_title('{}'.format(image_speed))
    if spd_ind == 0:
        plot_tools.addScaleBars(ax1[spd_ind], dT=2, dF=0.25, T_value=-0.1, F_value=-0.08)
fh1.suptitle('Background speed ($\degree$/s)')

fh1.savefig(os.path.join(save_directory, 'natimage_egfly_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)

# %% heatmaps for all gloms
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]
fh2, ax2 = plt.subplots(3, 3, figsize=(2.5, 3.5), tight_layout=True)
[x.set_axis_off() for x in ax2.ravel()]
cbar_ax = fh2.add_axes([.91, .3, .03, .4])
for g_ind, glom in enumerate(included_gloms):
    ax2[rows[g_ind], cols[g_ind]].set_axis_on()
    glom_data = ID.getResponseAmplitude(np.nanmean(all_resp_mat[g_ind, ...], axis=-1), metric='max')

    df = pd.DataFrame(data=glom_data, columns=image_speeds, index=[filter_list[x] for x in filter_codes])
    if g_ind == 5:
        xticklabels = np.array(image_speeds).astype('int')
        yticklabels = [filter_list[x][0] for x in filter_codes]
    else:
        xticklabels = False
        yticklabels = False

    if g_ind == 0:
        cbar = True
        cbar_ax = cbar_ax
        cbar_kws={'label': 'Response (dF/F)'}

    else:
        cbar=False
    sns.heatmap(df.T, ax=ax2[rows[g_ind], cols[g_ind]],
                xticklabels=xticklabels, yticklabels=yticklabels,
                vmin=0, vmax=0.35, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws,
                rasterized=True, cmap='viridis')
    ax2[rows[g_ind], cols[g_ind]].set_title(glom, color=util.get_color_dict()[glom], fontsize=10, fontweight='bold')

ax2[2, 0].set_xlabel('Speed ($\degree$/sec)')

fh2.savefig(os.path.join(save_directory, 'natimage_tuning.svg'), transparent=True)


# %% Get filtered images for fig. panels
image_names = np.unique([x.replace('whitened_', '') for x in np.unique(np.array(ID.getEpochParameters('stim0_image_name')))])

pixels_per_degree = 1536 / 360
screen_width = 160 * pixels_per_degree  # deg -> pixels
screen_height = 50 * pixels_per_degree  # deg -> pixels


new_im = image.Image(image_name=image_names[2])
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

# %% Show example filtered images, and spectra

spectra = pd.read_pickle(os.path.join(dataio.get_config_file()['save_directory'], 'vh_images_meanspsectra.pkl'))
freq = spectra.columns.values
scale = 3e7
p_ideal = scale * 1 / freq**2

fh3, ax3 = plt.subplots(4, 2, figsize=(3, 3.75))
[x.set_axis_off() for x in ax3[:, 1]]
# Crop to about the extent of the image on the screen
pixels_per_degree = 1536 / 360
screen_width = 160 * pixels_per_degree  # deg -> pixels
screen_height = 50 * pixels_per_degree  # deg -> pixels
image_width = img_orig.shape[1]
image_height = img_orig.shape[0]
[x.set_xlim([image_width/2 - screen_width/2, image_width/2 + screen_width/2]) for x in ax3[:, 1]]
[x.set_ylim([image_height/2 - image_height/2, image_height/2 + image_height/2]) for x in ax3[:, 1]]
[x.set_xlim([1e-3, 1e-1]) for x in ax3[:, 0]]
[x.set_ylim([1e9, 1e13]) for x in ax3[:, 0]]
[x.loglog(freq, p_ideal, 'k', alpha=0.5) for x in ax3[:, 0]]


ax3[0, 1].imshow(img_orig, cmap='Greys_r')
ax3[0, 0].loglog(freq, spectra.loc['original', :].values, label='original')
ax3[0, 0].annotate('Original', (1.2e-3, 3e9))

ax3[1, 1].imshow(img_white, cmap='Greys_r')
ax3[1, 0].loglog(freq, spectra.loc['white', :].values)
ax3[1, 0].annotate('Whitened', (1.2e-3, 3e9))

ax3[2, 1].imshow(img_hp, cmap='Greys_r')
ax3[2, 0].loglog(freq, spectra.loc['highpass', :].values)
ax3[2, 0].annotate('Highpass', (1.2e-3, 3e9))

ax3[3, 1].imshow(img_lp, cmap='Greys_r')
ax3[3, 0].loglog(freq, spectra.loc['lowpass', :].values)
ax3[3, 0].annotate('Lowpass', (1.2e-3, 3e9))

[x.set_xticks([]) for x in ax3[:-1, 0]]
[x.set_yticks([]) for x in ax3[:-1, 0]]

ax3[3, 0].set_xlabel('Spatial freq. (cpd)')

fh3.supylabel('Power')
fh3.savefig(os.path.join(save_directory, 'natimage_spectra.svg'), transparent=True)


# %% (2) 8 DIRECTIONS
target_angles = [0, 45, 90, 135, 180, 225, 270, 315]
eg_series = ('2022-05-24', 6)  # ('2022-05-26', 2)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          'image_speed': [160],
                                                                          'background_direction': target_angles,
                                                                          'image_index': [0, 5, 15],
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
                                                                   query={'current_background_direction': ang},
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

    if True:  # plot individual fly responses, QC
        fh0, ax0 = plt.subplots(len(included_gloms), len(target_angles), figsize=(3, 4))
        fh0.suptitle('{}: {}'.format(file_name, series_number))
        [plot_tools.cleanAxes(x) for x in ax0.ravel()]
        [x.set_ylim([-0.1, 0.3]) for x in ax0.ravel()]
        for ang_ind, ang in enumerate(target_angles):
            ax0[g_ind, ang_ind].set_title('{}$\degree$'.format(ang), rotation=45)
            for g_ind, glom in enumerate(included_gloms):
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 1, :], color='k', linewidth=1, label='Nonbehaving' if (g_ind+ang_ind == 0) else None)
                ax0[g_ind, ang_ind].plot(trial_averages[g_ind, ang_ind, 0,  :], color='b', linewidth=1, label='Behaving' if (g_ind+ang_ind == 0) else None)

            # if g_ind == 0:
            #     ax0[gr_ind, gp_ind].set_title('{:.0f}'.format(ang))

# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, coh, dir, time, fly)
response_amps = np.stack(response_amps, axis=-1)  # (glom, image, speed, filter, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, rate, period, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, grate, period, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, grate, period, time)
# %% eg fly/glom

eg_fly_ind = 2
eg_glom_ind = 0
fh0, ax0 = plt.subplots(1, len(target_angles), figsize=(5, 1.5))
fh0.suptitle('Background direction')
[plot_tools.cleanAxes(x) for x in ax0.ravel()]
[x.set_ylim([-0.1, 0.4]) for x in ax0.ravel()]

for ang_ind, ang in enumerate(target_angles):
    ax0[ang_ind].set_title('{}$\degree$'.format(ang), rotation=0)
    ax0[ang_ind].plot(all_responses[eg_glom_ind, ang_ind, 1, :, eg_fly_ind],
                      color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2,
                      label='Nonbehaving' if (ang_ind == 0) else None)
    ax0[ang_ind].plot(all_responses[eg_glom_ind, ang_ind, 0,  :, eg_fly_ind],
                      color=util.get_color_dict()[included_gloms[eg_glom_ind]], linewidth=2, alpha=0.5,
                      label='Behaving' if (ang_ind == 0) else None)

fh0.legend()
fh0.savefig(os.path.join(save_directory, 'natimage_direction_egfly_{}.svg'.format(included_gloms[eg_glom_ind])), transparent=True)

# %% Summary: polar plots
rows = [0, 0, 0, 1, 1, 2, 2, 2]
cols = [0, 1, 2, 0, 1, 0, 1, 2]

fh1, ax1 = plt.subplots(3, 3, figsize=(4, 4), subplot_kw={'projection': 'polar'})
[x.set_axis_off() for x in ax1.ravel()]

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

    mean_tuning.append(np.stack(fly_tuning, axis=-1))

mean_tuning = np.stack(mean_tuning, axis=-1)
mean_tuning = np.nanmean(mean_tuning, axis=-1) # glom x dir x beh/nonbeh

# Plot mean dir tuning across flies, for beh vs. nonbeh trials
for g_ind, glom in enumerate(included_gloms):
    ax1[rows[g_ind], cols[g_ind]].set_axis_on()
    plot_dir = np.append(target_angles, target_angles[0])
    # Behaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 0], mean_tuning[g_ind, 0, 0])
    ax1[rows[g_ind], cols[g_ind]].plot(np.deg2rad(plot_dir), plot_resp,
                    color=util.get_color_dict()[glom], linewidth=2, marker='.', alpha=1,
                    label='Behaving' if (g_ind == 0) else None, linestyle=':')

    # Nonbehaving trials
    plot_resp = np.append(mean_tuning[g_ind, :, 1], mean_tuning[g_ind, 0, 1])
    ax1[rows[g_ind], cols[g_ind]].plot(np.deg2rad(plot_dir), plot_resp,
                    color=util.get_color_dict()[glom], linewidth=2, marker='.',
                    label='Nonbehaving' if (g_ind == 0) else None, linestyle='-')




    if g_ind == 0:
        pass
    else:
        ax1[rows[g_ind], cols[g_ind]].set_xticklabels([])
        # ax1[rows[g_ind], cols[g_ind]].set_yticks([0.1, 0.2])
        # ax1[rows[g_ind], cols[g_ind]].set_ylim([0, 0.25])

    y_lim = np.ceil(np.max(mean_tuning[g_ind, ...]) / 0.1) * 0.1

    ax1[rows[g_ind], cols[g_ind]].set_ylim([0, y_lim])
    ax1[rows[g_ind], cols[g_ind]].set_yticks([y_lim])

    ax1[rows[g_ind], cols[g_ind]].annotate(glom, (np.deg2rad(160), 0.7*y_lim),
                                           ha='center', rotation=0, fontsize=9, color=util.get_color_dict()[glom])

fh1.legend()
fh1.savefig(os.path.join(save_directory, 'natimage_direction_polar.svg'), transparent=True)
