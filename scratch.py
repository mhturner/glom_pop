



# %% Eg video of Gcamp resp and behavior...
import nibabel as nib
import os
from skimage.filters import gaussian
import numpy as np
import ants


eg_series = ('2022-04-12', 1)

data_dir = '/Users/mhturner/CurrentData/20220412'
ch2 = np.asarray(nib.load(os.path.join(data_dir, 'TSeries-20220412-001_reg.nii')).dataobj, dtype='uint16')[:, :, :, :, 1]
ch2.shape
ch2_smooth = gaussian(ch2, sigma=(1, 1, 0, 3))
ants.image_write(ants.from_numpy(ch2_smooth), os.path.join(data_dir, 'TSeries-20220412-001_smooth_ch2.nii'))
# %% Fictrac...

import numpy as np
import os
import matplotlib.pyplot as plt
from visanalysis.analysis import imaging_data, shared_analysis
from glom_pop import dataio, util, fictrac
import glob
import pandas as pd
from scipy.signal import resample, savgol_filter
from visanalysis.util import plot_tools

file_name = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/datafiles/2022-04-12'
series_number = 1
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)
file_path = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync/datafiles/2022-04-12.hdf5'

file_name = os.path.split(file_name)[-1]
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=True)

# Get behavior data
behavior_data = dataio.load_behavior(ID, process_behavior=True)

# Load response data
response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)
response_amp = ID.getResponseAmplitude(epoch_response_matrix, metric='max')

# FT data:
ft_dir = '/Users/mhturner/CurrentData/20220412/'
dir = os.path.join(ft_dir, 'series001')
filename = os.path.split(glob.glob(os.path.join(dir, '*.dat'))[0])[-1]

ft_data = pd.read_csv(os.path.join(dir, filename), header=None)
sphere_radius = 4.5e-3 # in m
fps = 50  # hz

frame = ft_data.iloc[:, 0]
xrot = ft_data.iloc[:, 5]
yrot = ft_data.iloc[:, 6] * sphere_radius * fps * 1000 # fwd --> in mm/sec
zrot = ft_data.iloc[:, 7]  * 180 / np.pi * fps # rot  --> deg/sec

heading = ft_data.iloc[:, 16]
direction = ft_data.iloc[:, 16] + ft_data.iloc[:, 17]

speed = ft_data.iloc[:, 18]

x_loc = ft_data.iloc[:, 14]
y_loc = ft_data.iloc[:, 15]


xrot_filt = savgol_filter(xrot, 41, 3)
yrot_filt = savgol_filter(yrot, 41, 3)
zrot_filt = savgol_filter(zrot, 41, 3)

zrot_ds = resample(zrot_filt, response_data.get('response').shape[1])
_, turning_response_matrix = ID.getEpochResponseMatrix(zrot_ds[np.newaxis, :],
                                                       dff=False)

turning_amp = ID.getResponseAmplitude(turning_response_matrix, metric='mean')
new_turning_corr = np.array([np.corrcoef(turning_amp, response_amp[x, :])[0, 1] for x in range(len(included_gloms))])
new_turning_corr

plt.plot(turning_amp[0, :], response_amp[0, :], 'ko')

# plt.plot(x_loc, y_loc, 'k-')
fh0, ax0 = plt.subplots(1, 1, figsize=(12, 3))
ax0.plot(zrot_ds)
# %%
eg_trials = np.arange(0, 100)
y_min = -0.15
y_max = 0.80

concat_response = np.concatenate([epoch_response_matrix[:, x, :] for x in eg_trials], axis=1)
concat_running = np.concatenate([behavior_data.get('running_response_matrix')[:, x, :] for x in eg_trials], axis=1)
concat_turning = np.concatenate([turning_response_matrix[:, x, :] for x in eg_trials], axis=1)
concat_behaving = np.concatenate([behavior_data.get('behavior_binary_matrix')[:, x, :] for x in eg_trials], axis=1)
concat_time = np.arange(0, concat_running.shape[1]) * ID.getAcquisitionMetadata('sample_period')

# Red triangles when stim hits center of screen (middle of trial)
dt = np.diff(concat_time)[0]  # sec
trial_len = epoch_response_matrix.shape[2]
concat_len = len(concat_time)
y_val = 0.5

fh0, ax0 = plt.subplots(3+len(included_gloms), 1, figsize=(18, 8))
[x.set_ylim([y_min, y_max]) for x in ax0.ravel()]
[util.clean_axes(x) for x in ax0.ravel()]
[x.set_ylim() for x in ax0.ravel()]
ax0[0].plot(dt * np.linspace(trial_len/2,
                             concat_len-trial_len/2,
                             len(eg_trials)),
            y_val * np.ones(len(eg_trials)),
            'rv', markersize=4)
ax0[0].set_ylim([0.25, 0.75])
ax0[0].plot(concat_time, np.zeros_like(concat_time), color='w')
ax0[0].set_ylabel('Stim', rotation=0)

ax0[1].plot(concat_time, concat_running[0, :], color='k')
ax0[1].set_ylim([concat_running.min(), concat_running.max()])
ax0[1].set_ylabel('Movement', rotation=0)

ax0[2].plot(concat_time, concat_turning[0, :], color='k')
ax0[2].set_ylim([concat_turning.min(), concat_turning.max()])
ax0[2].set_ylabel('Rotation', rotation=0)
plot_tools.addScaleBars(ax0[2], dT=4, dF=200, T_value=-2, F_value=-100)

for g_ind, glom in enumerate(included_gloms):
    ax0[3+g_ind].set_ylabel(glom, rotation=0)
    ax0[3+g_ind].fill_between(concat_time, concat_behaving[0, :], color='k', alpha=0.25, linewidth=0)
    ax0[3+g_ind].plot(concat_time, concat_response[g_ind, :], color=util.get_color_dict()[glom])
    ax0[3+g_ind].set_ylim([concat_response.min(), concat_response.max()])
    if g_ind == 0:
        plot_tools.addScaleBars(ax0[3+g_ind], dT=4, dF=0.25, T_value=-1, F_value=-0.1)
