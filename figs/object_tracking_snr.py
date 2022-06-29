import numpy as np
import os
from glom_pop import dataio
from skimage.io import imread
import matplotlib.pyplot as plt

images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')

im_fn = 'imk00152.tif'
img = np.flipud(imread(os.path.join(images_dir, im_fn)))

# %%

# TODO: probe and background speed, same units.
# Continuous background speed as a fraction of spot speed
# Background speed relative to spot speed vs SNR or similar


t_steps = 800
image_speed_scale = 1  # fraction of rf radius
spot_speed = 2
spot_intensity = 0
ctr_y = 256   # for spot and rf
rf_ctr_x = int(img.shape[1]/2)  # for rf
rf_radius = 60
spot_radius = 30


np.random.seed(1)
trajectory = np.cumsum(np.random.normal(loc=0, scale=image_speed_scale*rf_radius, size=t_steps+1))


# TODO: circular patch
c_stationary = np.zeros(t_steps)
l_stationary = np.zeros(t_steps)
c_moving = np.zeros(t_steps)
l_moving = np.zeros(t_steps)

lum_image = np.zeros(t_steps)  # Image moves
con_image = np.zeros(t_steps)

lum_spot = np.zeros(t_steps)  # Spot moves
con_spot = np.zeros(t_steps)

lum_both = np.zeros(t_steps)  # Image & spot both move
con_both = np.zeros(t_steps)

# Schematic of image, spot and rf
fh0, ax0 = plt.subplots(1, 1, figsize=(4, 4))
schem_spot_loc = 1400

img_tmp = img.copy()
img_tmp[(ctr_y-spot_radius):(ctr_y+spot_radius), (schem_spot_loc-spot_radius): (schem_spot_loc+spot_radius)] = spot_intensity
ax0.imshow(img_tmp, cmap='Greys_r')
circle1 = plt.Circle((rf_ctr_x, ctr_y), rf_radius, color='r', fill=False, linewidth=3)
ax0.add_patch(circle1)
ax0.set_axis_off()

for t in range(t_steps):
    ctr_x = spot_speed*t

    # (1) Image moves / no spot
    img_tmp = np.roll(img, shift=int(np.ceil(trajectory[t])), axis=1)
    local_pix = img_tmp[(ctr_y-rf_radius):(ctr_y+rf_radius), (rf_ctr_x-rf_radius):(rf_ctr_x+rf_radius)]
    con_image[t] = np.var(local_pix) / np.mean(local_pix)
    lum_image[t] = np.mean(local_pix)

    # (2) Spot moves / image stationary
    img_tmp = img.copy()
    img_tmp[(ctr_y-spot_radius):(ctr_y+spot_radius), (ctr_x-spot_radius): (ctr_x+spot_radius)] = spot_intensity
    local_pix = img_tmp[(ctr_y-rf_radius):(ctr_y+rf_radius), (rf_ctr_x-rf_radius):(rf_ctr_x+rf_radius)]
    con_spot[t] = np.var(local_pix) / np.mean(local_pix)
    lum_spot[t] = np.mean(local_pix)

    # (3) Spot & image both move
    img_tmp = np.roll(img, shift=int(np.ceil(trajectory[t])), axis=1)
    img_tmp[(ctr_y-spot_radius):(ctr_y+spot_radius), (ctr_x-spot_radius):(ctr_x+spot_radius)] = spot_intensity
    local_pix = img_tmp[(ctr_y-rf_radius):(ctr_y+rf_radius), (rf_ctr_x-rf_radius):(rf_ctr_x+rf_radius)]
    con_both[t] = np.var(local_pix) / np.mean(local_pix)
    lum_both[t] = np.mean(local_pix)


# %
fh, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(con_image, 'k', label='Moving background')
ax[0].plot(con_spot, 'b', label='Moving spot')
ax[0].plot(con_both, 'r', label='Spot & background move')
ax[0].set_ylabel('Local contrast')

ax[1].plot(lum_image, 'k')
ax[1].plot(lum_spot, 'b')
ax[1].plot(lum_both, 'r')
ax[1].set_ylabel('Local luminance')

ax[1].set_xlabel('Time')
fh.legend()
# %%




# %%
