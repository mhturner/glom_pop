<<<<<<< Updated upstream
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
from glom_pop import dataio, util
from scipy.stats import spearmanr
from scipy.interpolate import interp1d



dir = '/Users/mhturner/CurrentData'
fn = 'red_glom_traces_20220412_1.pkl'
sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
data_directory = os.path.join(sync_dir, 'datafiles')
ft_dir = os.path.join(sync_dir, 'behavior_tracking')


with open(os.path.join(dir, fn),'rb') as f:
    glom_responses = pkl.load(f)



# %%

file_path = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/datafiles/2022-04-12.hdf5'
series_number = 1
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)


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
# %%


# # Fictrac data:
ft_data_path = dataio.get_ft_datapath(ID, ft_dir)
behavior_data = dataio.load_fictrac_data(ID, ft_data_path,
                                         response_len=response_data.get('response').shape[1],
                                         process_behavior=True, fps=50, exclude_thresh=300)
trial_time = ID.getStimulusTiming(plot_trace_flag=False)['stimulus_start_times'] - ID.getRunParameters('pre_time')
f_interp_behaving = interp1d(trial_time, behavior_data.get('is_behaving')[0, :], kind='previous', bounds_error=False, fill_value=np.nan)


response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
new_beh_corr = np.array([spearmanr(behavior_data.get('walking_amp')[0, :], response_amp[x, :]).correlation for x in range(len(included_gloms))])


# %%

y_min = -0.15
y_max = 0.80
eg_trials = np.arange(30, 50)
# fh0: snippet of movement and glom response traces
fh0, ax0 = plt.subplots(2+len(included_gloms), 1, figsize=(5.5, 3.35))
# [x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
[util.clean_axes(x) for x in ax0.ravel()]
# [x.set_ylim() for x in ax0.ravel()]

concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
concat_walking = np.concatenate([behavior_data.get('walking_response_matrix')[:, x, :] for x in eg_trials], axis=1)
concat_time = np.arange(0, concat_walking.shape[1]) * ID.getAcquisitionMetadata('sample_period')
concat_behaving = f_interp_behaving(trial_time[eg_trials][0]+concat_time)

# Red triangles when stim hits center of screen (middle of trial)
dt = np.diff(concat_time)[0]  # sec
trial_len = epoch_response_matrix.shape[2]
concat_len = len(concat_time)
y_val = 0.5
ax0[0].plot(dt * np.linspace(trial_len/2,
                             concat_len-trial_len/2,
                             len(eg_trials)),
            y_val * np.ones(len(eg_trials)),
            'rv', markersize=4)
ax0[0].set_ylim([0.25, 0.75])
ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')

ax0[1].plot(concat_time, concat_walking[0, :], color='k')
ax0[1].set_ylim([concat_walking.min(), concat_walking.max()])
ax0[1].set_ylabel('Walking', rotation=0)

for g_ind, glom in enumerate(included_gloms):
    ax0[2+g_ind].set_ylabel(glom, rotation=0)
    ax0[2+g_ind].fill_between(concat_time, concat_behaving, color='k', alpha=0.25, linewidth=0)
    ax0[2+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
    # ax0[2+g_ind].set_ylim([concat_response.min(), concat_response.max()])
    if g_ind == 0:
        plot_tools.addScaleBars(ax0[2+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)

# fh2: overall movement trace, with threshold and classification shading
time_ds = ID.getAcquisitionMetadata('sample_period') * np.arange(response_data.get('response').shape[1])
fh2, ax2 = plt.subplots(1, 1, figsize=(4.5, 1.5))
ax2.set_ylabel('Walking amp.')
tw_ax = ax2.twinx()
tw_ax.fill_between(time_ds,
                   f_interp_behaving(time_ds),
                   color=[0.5, 0.5, 0.5], alpha=0.5, linewidth=0.0)
ax2.axhline(behavior_data.get('thresh'), color='r')
ax2.plot(time_ds, behavior_data.get('walking_mag_ds'),
                 color='k')
tw_ax.set_yticks([])

ax2.set_xlabel('Time (s)')
# %%





# %%
=======
import nump as np
>>>>>>> Stashed changes
