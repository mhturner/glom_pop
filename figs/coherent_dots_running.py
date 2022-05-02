from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
from scipy.signal import resample, savgol_filter
from skimage import filters
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel, ttest_1samp

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)
eg_series = ('2022-03-24', 14)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'CoherentDots',
                                                                          'include_in_analysis': True,
                                                                          },
                                                  target_groups=['aligned_response', 'behavior'],
                                                  )
# %%
target_coherence = [0, 0.25, 0.5, 0.75, 1.0]
target_speed = 80
eg_ind = 2

all_responses = []
response_amplitudes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response and behavior data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    behavior_data = dataio.load_behavior(ID)
    # downsample behavior from video rate (50 Hz) to imaging rate (~8 Hz)
    rmse_ds = resample(behavior_data['rmse'], response_data.get('response').shape[1])
    # smooth behavior trace.
    #   Window size about 200 msec, 1st order polynomial
    #   Keeps rapid onsets/offsets pretty well but makes bouts more obvious and continuous
    rmse_ds = savgol_filter(rmse_ds, 9, 1)

    thresh = filters.threshold_li(rmse_ds)
    binary_behavior_ds = (rmse_ds > thresh).astype('int')

    # Align running responses
    _, running_response_matrix = ID.getEpochResponseMatrix(rmse_ds[np.newaxis, :],
                                                           dff=False)
    _, behavior_binary_matrix = ID.getEpochResponseMatrix(binary_behavior_ds[np.newaxis, :],
                                                          dff=False)

    # Categorize trial as behaving or nonbehaving
    beh_per_trial = np.mean(behavior_binary_matrix[0, :, :], axis=1)
    behaving = beh_per_trial > 0.5  # bool array: n trials
    behaving_trials = np.where(behaving)[0]
    nonbehaving_trials = np.where(~behaving)[0]
    fh, ax = plt.subplots(2, 1, figsize=(6, 3))
    ax[0].plot(rmse_ds, 'k')
    ax[1].plot(binary_behavior_ds, 'r')

    # Align and sort response data
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Select only trials with target params:
    # Shape =  (gloms x param conditions x behaving x time)
    trial_averages = np.zeros((len(included_gloms), len(target_coherence), 2, epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    num_matching_trials = []

    for coh_ind, coh in enumerate(target_coherence):
        _, inds = shared_analysis.filterTrials(epoch_response_matrix,
                                               ID,
                                               query={'coherence': coh,
                                                      'speed': target_speed},
                                               return_inds=True)
        behaving_inds = np.array([x for x in inds if x in behaving_trials])
        if len(behaving_inds) >= 1:
            trial_averages[:, coh_ind, 0, :] = np.nanmean(epoch_response_matrix[:, behaving_inds, :], axis=1)  # each trial average: gloms x time

        nonbehaving_inds = np.array([x for x in inds if x in nonbehaving_trials])
        if len(nonbehaving_inds) >= 1:
            trial_averages[:, coh_ind, 1, :] = np.nanmean(epoch_response_matrix[:, nonbehaving_inds, :], axis=1)  # each trial average: gloms x time

    if np.all(np.array(num_matching_trials) > 0):
        print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
        response_amp = ID.getResponseAmplitude(trial_averages, metric='mean')  # shape = gloms x param condition

        all_responses.append(trial_averages)
        response_amplitudes.append(response_amp)

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        # eg fly: show responses to 0 and 1 coherence
        fh0, ax = plt.subplots(2, len(included_gloms), figsize=(8, 3), gridspec_kw={'hspace': 0})
        [x.set_ylim([-0.15, 0.35]) for x in ax.ravel()]
        [x.set_xlim([-0.25, response_data['time_vector'].max()]) for x in ax.ravel()]
        [util.clean_axes(x) for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            ax[0, g_ind].set_title(glom, fontsize=9, rotation=45)
            for u_ind, up in enumerate([0.0, 1.0]):
                pull_ind = np.where(np.array(target_coherence) == up)[0][0]
                if u_ind == 0:
                    plot_tools.addScaleBars(ax[0, 0], dT=4, dF=0.25, T_value=-0.1, F_value=-0.1)

                ax[u_ind, g_ind].plot(response_data['time_vector'],
                                      trial_averages[g_ind, pull_ind, 1, :],
                                      color='k', alpha=0.75)
                ax[u_ind, g_ind].plot(response_data['time_vector'],
                                      trial_averages[g_ind, pull_ind, 0, :],
                                      color=util.get_color_dict()[glom])



# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, behaving, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, behaving, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

# Are responses (across all flies) significantly different than zero?
p_sig_responses = np.array([ttest_1samp(all_responses.mean(axis=(1, 2, 3))[g_ind, :], 0)[1] for g_ind in range(len(included_gloms))])

# %%
response_amplitudes.shape

response_amplitudes.shape
norm_ra = response_amplitudes / response_amplitudes[:, 0, 0, :][:, np.newaxis, np.newaxis, ...]
norm_ra.shape

mean_norm_ra = np.nanmean(norm_ra, axis=-1)

sns.heatmap(mean_norm_ra[1, :, :], cmap='viridis')



# %%

fh, ax = plt.subplots(4, 4, figsize=(4, 4))
ax = ax.ravel()
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].plot(mean_norm_ra[g_ind, :, 0].T, 'k')
    ax[g_ind].plot(mean_norm_ra[g_ind, :, 1].T, 'b')





# %%
