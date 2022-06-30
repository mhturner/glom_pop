import numpy as np
import os
import glob
from glom_pop import dataio
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']

images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')

fns = np.sort(glob.glob(os.path.join(images_dir, '*.tif')))

# %%
# Image dims are: 1536 x 512 pixels
#       = 360 x 120 degrees
# Rescale to 1 deg/pixel on loading img
# pixels_per_degree = 1536 / 360
time_vec = np.arange(0, 4, 0.1)
spot_speed = 100  # deg/sec
# image_speeds = [20, 40, 80, 160, 320]  # deg/sec
image_speeds = np.arange(20, 320, 40)  # deg/sec
spot_contrast = -0.5
ctr_y = 60   # for spot and rf deg
rf_ctr_x = 180  # deg, for rf

rf_radius = 20  # deg
spot_radius = 7.5  # deg


def get_d_prime(spot_present, spot_absent):
    d_prime = (np.mean(spot_present) - np.mean(spot_absent)) / np.sqrt((np.var(spot_present) + np.var(spot_absent))/2)
    return d_prime


dprime_lum = np.empty((len(fns), len(image_speeds)))
dprime_con = np.empty((len(fns), len(image_speeds)))

for im_ind, im_fn in enumerate(fns):
    img = resize(np.flipud(imread(im_fn)), (120, 360))
    img = img / np.max(img)
    spot_intensity = np.mean(img) + spot_contrast * np.mean(img)

    for spd_ind, image_speed in enumerate(image_speeds):
        lum_image = np.zeros(len(time_vec))  # Image moves
        con_image = np.zeros(len(time_vec))

        lum_spot = np.zeros(len(time_vec))  # Spot moves
        con_spot = np.zeros(len(time_vec))

        lum_both = np.zeros(len(time_vec))  # Image & spot both move
        con_both = np.zeros(len(time_vec))

        for t_ind, t in enumerate(time_vec):
            ctr_x = spot_speed * t
            ctr_image = image_speed * t

            # (1) Image moves / no spot
            img_tmp = np.roll(img, shift=int(ctr_image), axis=1)
            local_pix = img_tmp[int(ctr_y-rf_radius):int(ctr_y+rf_radius), int(rf_ctr_x-rf_radius):int(rf_ctr_x+rf_radius)]
            con_image[t_ind] = np.var(local_pix) / np.mean(local_pix)
            lum_image[t_ind] = np.mean(local_pix)

            # (2) Spot moves / image stationary
            img_tmp = img.copy()
            img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(ctr_x-spot_radius): int(ctr_x+spot_radius)] = spot_intensity
            local_pix = img_tmp[int(ctr_y-rf_radius):int(ctr_y+rf_radius), int(rf_ctr_x-rf_radius):int(rf_ctr_x+rf_radius)]
            con_spot[t_ind] = np.var(local_pix) / np.mean(local_pix)
            lum_spot[t_ind] = np.mean(local_pix)

            # (3) Spot & image both move
            img_tmp = np.roll(img, shift=int(ctr_image), axis=1)
            img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(ctr_x-spot_radius):int(ctr_x+spot_radius)] = spot_intensity
            local_pix = img_tmp[int(ctr_y-rf_radius):int(ctr_y+rf_radius), int(rf_ctr_x-rf_radius):int(rf_ctr_x+rf_radius)]
            con_both[t_ind] = np.var(local_pix) / np.mean(local_pix)
            lum_both[t_ind] = np.mean(local_pix)

        # Window of time when spot is in RF
        window_st = (rf_ctr_x-rf_radius-spot_radius) / spot_speed  # sec
        window_end = (rf_ctr_x+rf_radius+spot_radius) / spot_speed  # sec
        # Index in time vector for window start and stop
        window_st_ind = np.where(time_vec > window_st)[0][0]
        window_end_ind = np.where(time_vec < window_end)[0][-1]

        # D prime for spot detection
        # Note sign flip for luminance b/c a "hit" is associated with lower luminance (dark spot)
        dprime_lum[im_ind, spd_ind] = -1 * get_d_prime(spot_present=lum_both[window_st_ind:window_end_ind],
                                                       spot_absent=lum_image[window_st_ind:window_end_ind])

        dprime_con[im_ind, spd_ind] = get_d_prime(spot_present=con_both[window_st_ind:window_end_ind],
                                                  spot_absent=con_image[window_st_ind:window_end_ind])

        if np.logical_and(image_speed == 60, im_ind == 6):  # Eg traces plot
            # Schematic of image, spot and rf
            fh0, ax0 = plt.subplots(1, 1, figsize=(4, 1.5))
            schem_spot_loc = 180

            img_tmp = img.copy()
            img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(schem_spot_loc-spot_radius):int(schem_spot_loc+spot_radius)] = spot_intensity
            ax0.imshow(img_tmp, cmap='Greys_r')
            circle1 = plt.Circle((rf_ctr_x, ctr_y), rf_radius, color=[1, 1, 1], fill=False, linewidth=2, linestyle='--')
            ax0.add_patch(circle1)
            ax0.set_axis_off()

            # Spot on static background
            fh1, ax1 = plt.subplots(1, 2, figsize=(4, 2), tight_layout=True)
            [x.spines['top'].set_visible(False) for x in ax1.ravel()]
            [x.spines['right'].set_visible(False) for x in ax1.ravel()]
            [x.axvline(window_st, color='k', alpha=0.5, linestyle='--') for x in ax1.ravel()]
            [x.axvline(window_end, color='k', alpha=0.5, linestyle='--') for x in ax1.ravel()]
            ax1[0].plot(time_vec, lum_spot, color='k', linewidth=2)
            ax1[0].set_ylim([lum_both.min(), lum_both.max()])
            ax1[0].set_ylabel('Luminance')
            ax1[1].plot(time_vec, con_spot, color='k', linewidth=2)
            ax1[1].set_ylabel('Spatial contrast')
            ax1[1].set_ylim([con_both.min(), con_both.max()])
            fh1.suptitle('Spot on static background')

            # Moving background
            fh2, ax2 = plt.subplots(1, 2, figsize=(4, 2), tight_layout=True)
            [x.spines['top'].set_visible(False) for x in ax2.ravel()]
            [x.spines['right'].set_visible(False) for x in ax2.ravel()]
            [x.axvline(window_st, color='k', alpha=0.5, linestyle='--') for x in ax2.ravel()]
            [x.axvline(window_end, color='k', alpha=0.5, linestyle='--') for x in ax2.ravel()]
            ax2[0].plot(time_vec, lum_image, color=[0.5, 0.5, 0.5], label='Background alone', linewidth=2)
            ax2[0].plot(time_vec, lum_both, color=sns.desaturate('b', 0.5), label='Background + spot', linewidth=2)
            ax2[0].set_ylim([lum_both.min(), lum_both.max()])
            ax2[0].set_ylabel('Luminance')
            ax2[0].set_xlabel('Time (s)')

            ax2[1].plot(time_vec, con_image, color=[0.5, 0.5, 0.5], linewidth=2)
            ax2[1].plot(time_vec, con_both, color=sns.desaturate('b', 0.5), linewidth=2)
            ax2[1].set_ylim([con_both.min(), con_both.max()])
            ax2[1].set_ylabel('Spatial contrast')
            ax2[1].set_xlabel('Time (s)')

            [x.spines['top'].set_visible(False) for x in ax2.ravel()]
            [x.spines['right'].set_visible(False) for x in ax2.ravel()]
            fh2.legend()
            fh2.suptitle('Moving background')

fh0.savefig(os.path.join(save_directory, 'object_tracking_schem.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'object_tracking_trace_spot.svg'), transparent=True)
fh2.savefig(os.path.join(save_directory, 'object_tracking_trace_bg.svg'), transparent=True)
# %%
fh4, ax4 = plt.subplots(1, 2, figsize=(4, 2), tight_layout=True)
[x.spines['top'].set_visible(False) for x in ax4.ravel()]
[x.spines['right'].set_visible(False) for x in ax4.ravel()]
[x.set_ylim([0, 3.8]) for x in ax4]
[x.set_xlim([0, np.max(image_speeds)+20]) for x in ax4]
ax4[0].errorbar(x=image_speeds,
                y=np.mean(dprime_lum, axis=0),
                yerr=np.std(dprime_lum, axis=0) / np.sqrt(dprime_lum.shape[0]),
                color='k', marker='o')
ax4[0].set_title('Luminance')

ax4[1].errorbar(x=image_speeds,
                y=np.mean(dprime_con, axis=0),
                yerr=np.std(dprime_con, axis=0) / np.sqrt(dprime_con.shape[0]),
                color='k', marker='o')
ax4[1].set_title('Contrast')

ax4[1].plot(image_speeds, np.mean(dprime_con, axis=0), color='g', marker='o')

fh4.supxlabel('Image speed ($\degree$/sec)')
ax4[0].set_ylabel('Disrciminability ($d\'$)')

fh4.savefig(os.path.join(save_directory, 'object_tracking_dprime.svg'), transparent=True)



# %%
