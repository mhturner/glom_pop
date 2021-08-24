"""
Align template brain & glom map to meanbrain


maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import os
import numpy as np
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import pandas as pd
import colorcet as cc

from glom_pop import dataio

# %%

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')

mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format('20210824')
# %% LOAD
# Load meanbrain
meanbrain = ants.image_read(os.path.join(base_dir, 'anatomical_brains', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'anatomical_brains', mask_fn)).dataobj).astype('uint32')
lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain_red.spacing)

# Load template & glom mask
template = ants.image_read(os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %% SHOW TEMPLATE BRAIN
template_smoothed = ants.smooth_image(template, sigma=[3, 3, 2], sigma_in_physical_coordinates=False)

fh, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(18):
    ax[z].imshow(template_smoothed[:, :, z*5].T)

# %% SHOW MEAN BRAIN - GREEN. Align on this.

meanbrain_smoothed = ants.smooth_image(meanbrain_green, sigma=[2, 2, 1], sigma_in_physical_coordinates=False)

disp_masked = np.ma.masked_where(lobe_mask.numpy() == 0, meanbrain_smoothed.numpy())  # mask at 0

fh, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(18):
    ax[z].imshow(disp_masked[:, :, z*2].T)
# %% COMPUTE ALIGNMENT - SyN

reg = ants.registration(meanbrain_smoothed,  # fixed = meanbrain, green channel
                        template_smoothed,  # Moving = smoothed template
                        type_of_transform='ElasticSyN',
                        mask=lobe_mask,  # mask includes only gloms, excludes lobe
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=1)

# Save transform
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')
os.makedirs(transform_dir, exist_ok=True)
dataio.save_transforms(reg, transform_dir)

# Apply alignment to brain density and mask
transform_list = dataio.get_transform_list(transform_dir, direction='forward')

# APPLY ALIGNMENT TO MASK & TEMPLATE
template_transformed = ants.apply_transforms(fixed=meanbrain_green,
                                             moving=template,
                                             transformlist=transform_list,
                                             interpolator='nearestNeighbor')

glom_mask_transformed = ants.apply_transforms(fixed=meanbrain_green,
                                              moving=glom_mask,
                                              transformlist=transform_list,
                                              interpolator='genericLabel')

ants.image_write(template_transformed, os.path.join(transform_dir, 'JRC2018_reg2meanbrain.nii'))
ants.image_write(glom_mask_transformed, os.path.join(transform_dir, 'glom_mask_reg2meanbrain.nii'))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_transformed, glom_mask_transformed])
save_path = os.path.join(transform_dir, 'overlay_meanbrain.nii')
ants.image_write(merged, save_path)
# %% CHECK OVERALL ALIGNMENT
fh, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(meanbrain_green.max(axis=2).T)
ax[1].imshow(template_transformed.max(axis=2).T)

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='-', color='r')

# %%
# disp_meanbrain = meanbrain_red[55:230, 35:175, :]
# disp_glom = glom_mask_transformed[55:230, 35:175, :]

disp_meanbrain = meanbrain_red[:, :, :]
disp_glom = glom_mask_transformed[:, :, :]

glom_tmp = np.ma.masked_where(disp_glom == 0, disp_glom)  # mask at 0

slices = [5, 10, 20, 30, 40, 44]
fh, ax = plt.subplots(len(slices), 3, figsize=(12, len(slices)*3))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, slice in enumerate(slices):
    ax[s_ind, 0].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))

    ax[s_ind, 1].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, interpolation='nearest')

    ax[s_ind, 2].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))
    ax[s_ind, 2].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, interpolation='nearest')
