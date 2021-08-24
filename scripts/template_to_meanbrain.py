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
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from glom_pop import dataio

# %%

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')

mask_fn = 'lobe_mask_chat_meanbrain_{}_ch1.nii'.format('20210816')
# %% LOAD
# Load meanbrain
meanbrain = ants.image_read(os.path.join(base_dir, 'anatomical_brains', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brains', mask_fn)).dataobj).astype('uint32')
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

# %% CHECK OVERALL ALIGNMENT
fh, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(meanbrain_green.max(axis=2).T)
ax[1].imshow(template_transformed.max(axis=2).T)

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='-', color='r')

# %%
# Show z slices of meanbrain, template, & glom map for alignment
z_levels = [5, 10, 15, 20, 25, 30, 35, 40, 44]

glom_size_threshold = 300

vals = vpn_types.get('Unnamed: 0').values
names = vpn_types['vpn_types'].values

# mask out gloms with fewer than glom_size_threshold voxels
vox_per_type = []
for m_ind, mask_id in enumerate(vals):
    voxels_in_mask = np.sum(glom_mask_transformed.numpy() == mask_id)
    vox_per_type.append(voxels_in_mask)
    if voxels_in_mask < glom_size_threshold:
        glom_mask_transformed[glom_mask_transformed == mask_id] = 0
    else:
        pass

cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(glom_mask_transformed.numpy() == 0, glom_mask_transformed.numpy())  # mask at 0
grn_tmp = np.ma.masked_where(lobe_mask.numpy() == 0, meanbrain_green.numpy())  # mask at 0

fh, ax = plt.subplots(len(z_levels), 4, figsize=(8, 12))
[x.set_xticklabels([]) for x in ax.ravel()]
[x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

for z_ind, z in enumerate(z_levels):
    ax[z_ind, 0].imshow(meanbrain_red[:, :, z].T, cmap='Reds')
    ax[z_ind, 1].imshow(grn_tmp[:, :, z].T, cmap='Greens')
    ax[z_ind, 2].imshow(template_transformed[:, :, z].T, cmap='Blues')
    ax[z_ind, 3].imshow(glom_tmp[:, :, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind == 0:
        ax[z_ind, 0].set_title('mtdTomato')
        ax[z_ind, 1].set_title('syt1GCaMP6F')
        ax[z_ind, 2].set_title('JRC2018')
        ax[z_ind, 3].set_title('Glomerulus map')

        dx = 25 / 1.0  # um -> pix
        ax[z_ind, 0].plot([45, 45+dx], [180, 180], color='k', linestyle='-', marker='None', linewidth=2)
        ax[z_ind, 0].annotate('25 um', (40, 170))

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.set_xlim([35, 220])
    x.set_ylim([190, 20])


handles = [Patch(facecolor=color) for color in colors]

handles = np.array(handles)[np.array(vox_per_type)>glom_size_threshold]
names = np.array(names)[np.array(vox_per_type)>glom_size_threshold]

fh.legend(handles, [label for label in names], fontsize=10, ncol=6, handleheight=1.0, labelspacing=0.05)


# %%
disp_meanbrain = meanbrain_green[55:230, 35:175, :]
disp_glom = glom_mask_transformed[55:230, 35:175, :]

glom_tmp = np.ma.masked_where(disp_glom == 0, disp_glom)  # mask at 0

slices = [5, 10, 20, 30, 40, 44]
fh, ax = plt.subplots(len(slices), 3, figsize=(12, len(slices)*3))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, slice in enumerate(slices):
    ax[s_ind, 0].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))

    ax[s_ind, 1].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, interpolation='nearest')

    ax[s_ind, 2].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))
    ax[s_ind, 2].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, interpolation='nearest')
