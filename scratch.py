<<<<<<< HEAD
=======
<<<<<<< Updated upstream
import pickle as pkl
import os
>>>>>>> c7fed02a002d50e1a317d4e1981244a5a4588d4e
import numpy as np
import os
import glob
from glom_pop import dataio
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import imageio


save_directory = '/Users/mhturner/Dropbox/ClandininLab/Presentations/2022_NeuroLabNight'

images_dir = os.path.join(dataio.get_config_file()['images_dir'], 'vh_tif')

fns = np.sort(glob.glob(os.path.join(images_dir, '*.tif')))

# %%
# Image dims are: 1536 x 512 pixels
#       = 360 x 120 degrees
# Rescale to 1 deg/pixel on loading img
# pixels_per_degree = 1536 / 360
fps = 60
time_vec = np.linspace(0, 4, 4*fps)
spot_speed = 100  # deg/sec
spot_contrast = -0.5
ctr_y = 60   # for spot

spot_radius = 2.5  # deg

im_fn = fns[7]
img = resize(np.flipud(imread(im_fn)), (120, 360))

plt.imshow(img, cmap='Greys_r')

# %%
img = img / np.max(img)
spot_intensity = np.mean(img) + spot_contrast * np.mean(img)

image_speed = 360  # deg/sec


spot_only = []
spot_on_static = []
spot_on_moving = []


for t_ind, t in enumerate(time_vec):
    ctr_x = spot_speed * t
    ctr_image = image_speed * t

    # (1) Spot only
    img_tmp = 0.5 * np.ones_like(img)
    img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(ctr_x-spot_radius): int(ctr_x+spot_radius)] = spot_intensity
    spot_only.append(img_tmp.T*255)


    # (2) Spot moves / image stationary
    img_tmp = img.copy()
    img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(ctr_x-spot_radius): int(ctr_x+spot_radius)] = spot_intensity
    spot_on_static.append(img_tmp.T*255)

    # (3) Spot & image both move
    img_tmp = np.roll(img.copy(), shift=int(ctr_image), axis=1)
    img_tmp[int(ctr_y-spot_radius):int(ctr_y+spot_radius), int(ctr_x-spot_radius): int(ctr_x+spot_radius)] = spot_intensity
    spot_on_moving.append(img_tmp.T*255)


spot_only = np.stack(spot_only, axis=-1).astype('uint8')
spot_on_static = np.stack(spot_on_static, axis=-1).astype('uint8')
spot_on_moving = np.stack(spot_on_moving, axis=-1).astype('uint8')
# %%
spot_only = np.swapaxes(spot_only, 0, -1)
spot_on_static = np.swapaxes(spot_on_static, 0, -1)
spot_on_moving = np.swapaxes(spot_on_moving, 0, -1)

imageio.mimwrite(os.path.join(save_directory, 'spot_only.mp4'), spot_only , fps=fps)
imageio.mimwrite(os.path.join(save_directory, 'spot_on_static.mp4'), spot_on_static , fps=fps)
imageio.mimwrite(os.path.join(save_directory, 'spot_on_moving.mp4'), spot_on_moving , fps=fps)


# %%
=======
import nump as np
>>>>>>> Stashed changes
