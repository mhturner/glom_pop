import numpy as np
import os
import glob
import pims
import matplotlib.pyplot as plt
from sewar.full_ref import rmse as sewar_rmse
from visanalysis.analysis import imaging_data, shared_analysis
from scipy.signal import resample
import pandas as pd


from glom_pop import dataio, util, alignment

# TODO: double check timing based on video triggers? Not sure that's necessary tho
video_dir = '/Users/mhturner/CurrentData/'


sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
data_directory = os.path.join(sync_dir, 'datafiles')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

glom_size_threshold = 10

series_number = 8
file_name = '2022-03-18.hdf5'
# For video:
series_dir = 'Series008'
date_dir = '20220318'

file_path = os.path.join(data_directory, file_name)
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

### LOAD VIDEO ###
video_dir = os.path.join(video_dir, date_dir, series_dir)

filepath = glob.glob(video_dir + "/*.avi")[0]  # should be just one .avi in there

whole_vid = pims.as_grey(pims.Video(filepath))
fh, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(whole_vid[100], cmap='Greys_r')

cropped_vid = pims.as_grey(pims.process.crop(pims.Video(filepath), ((90, 0), (10, 20), (0, 0)) ))
ax[1].imshow(cropped_vid[100], cmap='Greys_r')

err_rmse = [sewar_rmse(cropped_vid[f], cropped_vid[f+1]) for f in range(len(cropped_vid)-1)]

fh, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(err_rmse)
###

# %%
# Load response data
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

# Resample to imaging rate and plot

err_rmse_ds = resample(err_rmse, response_data.get('response').shape[1])  # DO this properly based on response

fh, ax = plt.subplots(2, 1, figsize=(18, 2))
ax[0].plot(err_rmse_ds, 'k')
ax[1].plot(response_data.get('response')[2])

# %%
# epoch_response_matrix: shape=(gloms, trials, time)
epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
epoch_response_matrix[:] = np.nan

for val_ind, included_val in enumerate(included_vals):
    new_glom_size = np.sum(response_data.get('mask') == included_val)

    if new_glom_size > glom_size_threshold:
        pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
        epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
    else:  # Exclude because this glom, in this fly, is too tiny
        pass

# Align responses
_, running_response_matrix = ID.getEpochResponseMatrix(err_rmse_ds[np.newaxis, :], dff=False)
eg_trials = np.arange(50, 100)

fh, ax = plt.subplots(1+len(included_gloms), len(eg_trials), figsize=(20, 6))
[x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
[x.set_ylim() for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    ax[1+g_ind, 0].set_ylabel(glom)
    for t_ind, t in enumerate(eg_trials):
        ax[1+g_ind, t_ind].plot(epoch_response_matrix[g_ind, t, :], color=util.get_color_dict()[glom])

        if g_ind == 0:
            ax[0, t_ind].plot(running_response_matrix[0, t, :], color='k')
            ax[0, t_ind].set_axis_off()
            ax[0, t_ind].set_ylim([err_rmse_ds.min(), 40])

# %% response - triggered walking?

# %% summary

response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')
running_amp = ID.getResponseAmplitude(running_response_matrix, metric='mean')

fh, ax = plt.subplots(1+len(included_gloms), 1, figsize=(6, 6))
ax[0].plot(running_amp.T, 'k-')
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind+1].set_ylabel(glom)
    ax[g_ind+1].plot(response_amp[g_ind, :], color=util.get_color_dict()[glom])

# %%
corr_with_running = [np.corrcoef(running_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))]
corr_with_running = pd.DataFrame(data=corr_with_running, index=included_gloms)


fh, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.axhline(y=0, color='k', alpha=0.5)
ax.plot(corr_with_running, 'ko')
ax.set_ylim([-1, 1])





# %%
