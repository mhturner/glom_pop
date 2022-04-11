from glom_pop import dataio
from visanalysis.analysis import imaging_data
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


sync_dir = dataio.get_config_file()['sync_dir']
data_directory = os.path.join(sync_dir, 'datafiles')
video_dir = os.path.join(sync_dir, 'behavior_videos')

# %%

crops = [
        ((120, 80), (150, 150), (0, 0)),
         ]


ds = ('20220404', 1)

fh_tmp, ax_tmp = plt.subplots(len(crops)+1, 3, figsize=(12, 6))
for c_ind, crop in enumerate(crops):
    series_number = ds[1]
    file_name = '{}-{}-{}.hdf5'.format(ds[0][:4], ds[0][4:6], ds[0][6:])

    # For video:
    series_dir = 'series' + str(series_number).zfill(3)
    date_dir = ds[0]
    file_path = os.path.join(data_directory, file_name)
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Get video data:
    # Timing command from voltage trace
    voltage_trace, _, voltage_sample_rate = ID.getVoltageData()
    frame_triggers = voltage_trace[0, :]  # First voltage trace is trigger out

    video_filepath = glob.glob(os.path.join(video_dir, date_dir, series_dir) + "/*.avi")[0]  # should be just one .avi in there
    video_results = dataio.get_ball_movement(video_filepath,
                                             frame_triggers,
                                             sample_rate=voltage_sample_rate,
                                             cropping=crop)

    # Show cropped ball and overall movement trace for QC
    ax_tmp[c_ind, 0].plot(video_results['frame_times'], video_results['rmse'], 'k')
    ax_tmp[c_ind, 1].hist(video_results['rmse'], 500)
    ax_tmp[c_ind, 1].set_yscale('log')
    ax_tmp[c_ind, 2].imshow(video_results['cropped_frame'], cmap='Greys_r')



# %%


fh, ax = plt.subplots(2, 1, figsize=(6, 3.5))
ax[0].plot(video_results['rmse'], 'k')

tw_ax = ax[1].twinx()
tw_ax.fill_between(np.arange(len(video_results['binary_behavior'])), video_results['binary_behavior'], color='k', alpha=0.5)

ax[1].plot(video_results['rmse'], 'b')
# ax[1].set_xlim([3800, 4200])
ax[1].axhline(video_results['binary_thresh'], color='r')






# %%
