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

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')
data_directory = os.path.join(sync_dir, 'datafiles')
video_dir = os.path.join(sync_dir, 'behavior_videos')
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

# 2022-03-24.hdf5 : 1, 7, 12, 16
# 2022-03-18.hdf5: 8
series_number = 1
file_name = '2022-03-24.hdf5'
# For video:
series_dir = 'series' + str(series_number).zfill(3)
date_dir = '20220324'

file_path = os.path.join(data_directory, file_name)
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

### LOAD VIDEO ###
video_dir = os.path.join(video_dir, date_dir, series_dir)

filepath = glob.glob(video_dir + "/*.avi")[0]  # should be just one .avi in there

glob.glob(video_dir + "/*.avi")
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
epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
# Resample to imaging rate and plot

err_rmse_ds = resample(err_rmse, response_data.get('response').shape[1])  # DO this properly based on response

fh, ax = plt.subplots(len(included_gloms) + 1, 1, figsize=(18, 8))
ax[0].plot(err_rmse_ds, 'k')
ax[0].set_ylabel('Fly movement')

for g_ind, glom in enumerate(included_gloms):
    pull_ind = np.where(response_data['mask_vals'] == included_vals[g_ind])[0][0]
    ax[1+g_ind].set_ylabel(glom)
    ax[1+g_ind].plot(response_data.get('response')[pull_ind])
# ax[1].plot(response_data.get('response')[2])

# %%

# Align running responses
_, running_response_matrix = ID.getEpochResponseMatrix(err_rmse_ds[np.newaxis, :], dff=False)
eg_trials = np.arange(0, 50)

concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in range(epoch_response_matrix.shape[1])], axis=1)
concat_running = np.concatenate([running_response_matrix[:, x, :] for x in range(running_response_matrix.shape[1])], axis=1)

fh, ax = plt.subplots(1+len(included_gloms), 1, figsize=(12, 8))
[x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
[x.set_ylim() for x in ax.ravel()]

ax[0].plot(concat_running[0, :], color='k')
ax[0].set_ylim([err_rmse_ds.min(), 40])
ax[0].set_ylabel('Movement', rotation=0)
for g_ind, glom in enumerate(included_gloms):
    ax[1+g_ind].set_ylabel(glom)
    ax[1+g_ind].plot(concat_response[g_ind, :], color=util.get_color_dict()[glom])
    ax[1+g_ind].set_ylim([-0.1, 0.8])

fh.savefig(os.path.join(save_directory, 'ems_behavior_eg.pdf'), transparent=True)

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
