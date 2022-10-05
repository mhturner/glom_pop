import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.stats import ttest_1samp, ttest_rel
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from visanalysis.util import plot_tools
from glom_pop import dataio, util
import pickle as pkl

PROTOCOL_ID = 'ExpandingMovingSpot'

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')

eg_series = ('2022-04-12', 1)  # ('2022-04-12', 1): good, punctuated movement bouts
target_series_metadata = {'protocol_ID': PROTOCOL_ID,
                          'include_in_analysis': True,
                          'diameter': 15.0,
                          }
y_min = -0.15
y_max = 0.80
eg_trials = np.arange(30, 50)

included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata=target_series_metadata,
                                                  target_groups=['aligned_response', 'behavior'])


# %%

corr_with_running = []

response_amps = []
walking_amps = []
all_behaving = []
all_is_behaving = []

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

    # Get red channel glom traces into epoch_response_matrix
    date_str = os.path.split(series['file_name'])[1].replace('-','')
    fn = 'red_glom_traces_{}_{}.pkl'.format(date_str, series_number)
    with open(os.path.join(sync_dir, 'red_channel_ctl', fn),'rb') as f:
        glom_responses = pkl.load(f)

    time_vector, response_matrix = ID.getEpochResponseMatrix(glom_responses, dff=True)

    # Filter erm by included vals
    glom_size_threshold=10
    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_matrix[val_ind, :, :] = response_matrix[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
    response_amps.append(response_amp)

    # # Fictrac data:
    ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
    behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                             process_behavior=True, exclude_thresh=300, show_qc=False)
    all_is_behaving.append(behavior_data.get('is_behaving')[0])
    walking_amps.append(behavior_data.get('walking_amp'))
    new_beh_corr = np.array([spearmanr(behavior_data.get('walking_amp')[0, :], response_amp[x, :]).correlation for x in range(len(included_gloms))])
    corr_with_running.append(new_beh_corr)

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
    # if True:
        behaving_trial_matrix = np.zeros_like(behavior_data.get('walking_response_matrix'))
        behaving_trial_matrix[behavior_data.get('is_behaving'), :] = 1


        # fh0: snippet of movement and glom response traces
        fh0, ax0 = plt.subplots(2+len(included_gloms), 1, figsize=(5.5, 3.35))
        [x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
        [util.clean_axes(x) for x in ax0.ravel()]

        concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
        concat_walking = np.concatenate([behavior_data.get('walking_response_matrix')[:, x, :] for x in eg_trials], axis=1)
        concat_time = np.arange(0, concat_walking.shape[1]) * ID.getAcquisitionMetadata('sample_period')
        concat_behaving = np.concatenate([behaving_trial_matrix[:, x, :] for x in eg_trials], axis=1)

        # Red triangles when stim hits center of screen (middle of trial)
        dt = np.diff(concat_time)[0]  # sec
        trial_len = epoch_response_matrix.shape[2]
        concat_len = len(concat_time)
        trial_time = dt * np.linspace(trial_len/2,
                                     concat_len-trial_len/2,
                                     len(eg_trials))
        y_val = 0.5
        ax0[0].plot(trial_time,
                    y_val * np.ones(len(eg_trials)),
                    'rv', markersize=4)
        ax0[0].set_ylim([0.25, 0.75])
        ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')

        ax0[1].plot(concat_time, concat_walking[0, :], color='k')
        ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
        ax0[1].set_ylabel('Walking', rotation=0)
        for g_ind, glom in enumerate(included_gloms):
            ax0[2+g_ind].set_ylabel(glom, rotation=0)
            ax0[2+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
            ax0[2+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
            # ax0[2+g_ind].set_ylim([concat_response.min(), concat_response.max()])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[2+g_ind], dT=4, dF=0.25, T_value=0, F_value=-0.1)

corr_with_running = np.vstack(corr_with_running)  # flies x gloms
walking_amps = np.vstack(walking_amps)  # flies x trials
response_amps = np.dstack(response_amps)  # gloms x trials x flies
all_is_behaving = np.stack(all_is_behaving, axis=-1)  # trials x flies

# fh0.savefig(os.path.join(save_directory, 'repeat_beh_red_ctl_resp.svg'.format(PROTOCOL_ID)), transparent=True)
# fh2.savefig(os.path.join(save_directory, 'repeat_beh_red_ctl_running.svg'.format(PROTOCOL_ID)), transparent=True)

# %% Summary plots

# For each fly: corr between trial amplitude and trial behavior amount
fh2, ax2 = plt.subplots(1, 1, figsize=(2, 2.6))
ax2.axvline(0, color='k', alpha=0.50)
ax2.set_xlim([-0.8, 0.8])
ax2.invert_yaxis()

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    t_result = ttest_1samp(corr_with_running[:, g_ind], popmean=0, nan_policy='omit')
    p_vals.append(t_result.pvalue)

    y_mean = np.nanmean(corr_with_running[:, g_ind])
    y_err = np.nanstd(corr_with_running[:, g_ind]) / np.sqrt(corr_with_running.shape[0])
    ax2.plot(corr_with_running[:, g_ind], g_ind * np.ones(corr_with_running.shape[0]),
             marker='.', color=util.get_color_dict()[glom], linestyle='none', alpha=0.5)

    ax2.plot(y_mean, g_ind,
             marker='o', color=util.get_color_dict()[glom])

    ax2.plot([y_mean-y_err, y_mean+y_err], [g_ind, g_ind],
             color=util.get_color_dict()[glom])

# Multiple comparisons test. Step down bonferroni
h, p_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='holm')
for g_ind, glom in enumerate(included_gloms):
    if h[g_ind]:
        ax2.annotate('*', (0.5, g_ind), fontsize=12)

ax2.set_yticks([])
ax2.set_xlabel(r'Corr. with behavior ($\rho$)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
# fh2.savefig(os.path.join(save_directory, 'repeat_beh_red_ctl_summary.svg'.format(PROTOCOL_ID)), transparent=True)
