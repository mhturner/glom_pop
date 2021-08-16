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
# os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "1"

# del os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']
# %%

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210816')

mask_fn = 'lobe_mask_chat_meanbrain_{}_ch1.nii'.format('20210816')
# %% LOAD
# Load meanbrain
reference_fn = 'ZSeries-20210804-001'
filepath = os.path.join(base_dir, 'anatomical_brains', reference_fn)
metadata = dataio.get_bruker_metadata(filepath + '.xml')

meanbrain_red = dataio.get_ants_brain(os.path.join(base_dir, 'mean_brains', meanbrain_fn), metadata, channel=0)
meanbrain_green = dataio.get_ants_brain(os.path.join(base_dir, 'mean_brains', meanbrain_fn), metadata, channel=1)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brains', mask_fn)).dataobj).astype('uint32')
lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain_red.spacing)

atlas_spacing = (0.38, 0.38, 0.38)  # um
template = np.squeeze(np.asanyarray(nib.load(os.path.join(base_dir, 'template_brain', 'jrc2018.nii')).dataobj).astype('uint32'))
glom_mask = np.squeeze(np.asanyarray(nib.load(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed_eroded.nii')).dataobj).astype('uint32'))

# Remove mask voxels == 1 (LC10, not in volume really, messes with synthmorph)
glom_mask[glom_mask==1] = 0

template = ants.from_numpy(template, spacing=atlas_spacing)
glom_mask = ants.from_numpy(glom_mask, spacing=atlas_spacing)

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %% SHOW TEMPLATE BRAIN
template_smoothed = dataio.get_smooth_brain(template, smoothing_sigma=[3, 3, 2])

fh, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(18):
    ax[z].imshow(template_smoothed[:, :, z*5].T)

# %% SHOW MEAN BRAIN
meanbrain_smoothed = dataio.get_smooth_brain(meanbrain_red, smoothing_sigma=[2, 2, 1])

fh, ax = plt.subplots(4, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(18):
    ax[z].imshow(meanbrain_smoothed[:, :, z*2].T)

# %% COMPUTE ALIGNMENT - affine only

reg = ants.registration(meanbrain_smoothed,  # fixed = meanbrain
                        template_smoothed,  # Moving = smoothed template
                        type_of_transform='Affine',
                        mask=lobe_mask,  # mask includes only gloms, excludes lobe
                        flow_sigma=6,
                        total_sigma=1,
                        random_seed=1)

# APPLY ALIGNMENT TO (RAW) MASK & TEMPLATE
template_transformed = ants.apply_transforms(meanbrain_red,
                                             template,
                                             reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')

glom_mask_transformed = ants.apply_transforms(meanbrain_red,
                                              glom_mask,
                                              reg['fwdtransforms'],
                                              interpolator='genericLabel')

# # Save transformed brains and mask
nib.save(nib.Nifti1Image(glom_mask_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'aligned', 'glom_mask_affine2meanbrain.nii'))
nib.save(nib.Nifti1Image(template_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'aligned', 'JRC2018_affine2meanbrain.nii'))

# %% COMPUTE ALIGNMENT - SyN

# TODO: make this more repeatable across runs?
#   os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] should be set to "1" as well, I think. But this slows everything down a lot
reg = ants.registration(meanbrain_smoothed,  # fixed = meanbrain
                        template_smoothed,  # Moving = smoothed template
                        type_of_transform='SyN',
                        mask=lobe_mask,  # mask includes only gloms, excludes lobe
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=1)
# TODO: save transforms and re-load from permanent dir in apply step

# APPLY ALIGNMENT TO MASK & TEMPLATE
template_transformed = ants.apply_transforms(meanbrain_red,
                                             template,
                                             reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')

glom_mask_transformed = ants.apply_transforms(meanbrain_red,
                                              glom_mask,
                                              reg['fwdtransforms'],
                                              interpolator='genericLabel')

# # Save transformed brains and mask
nib.save(nib.Nifti1Image(glom_mask_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'aligned', 'glom_mask_reg2meanbrain.nii'))
nib.save(nib.Nifti1Image(template_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'aligned', 'JRC2018_reg2meanbrain.nii'))

# %% CHECK OVERALL ALIGNMENT
fh, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(meanbrain_red.max(axis=2).T)
ax[1].imshow(template_transformed.max(axis=2).T)

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='-', color='r')

# %%
# Show z slices of meanbrain, template, & glom map for alignment
z_levels = [5, 10, 15, 20, 25, 30]

glom_size_threshold = 300

vals = vpn_types.get('Unnamed: 0').values
names = vpn_types['vpn_types'].values

# mask out gloms with fewer than glom_size_threshold voxels
vox_per_type = []
for m_ind, mask_id in enumerate(vals):
    voxels_in_mask = np.sum(glom_mask_transformed.numpy() == mask_id)
    vox_per_type.append(voxels_in_mask)
    if voxels_in_mask < glom_size_threshold:
        glom_mask_transformed[glom_mask_transformed==mask_id] = 0
    else:
         pass


cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(glom_mask_transformed.numpy()==0, glom_mask_transformed.numpy()) # mask at 0

fh, ax = plt.subplots(len(z_levels), 3, figsize=(6, 12))
[x.set_xticklabels([]) for x in ax.ravel()]
[x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

for z_ind, z in enumerate(z_levels):
    ax[z_ind, 0].imshow(meanbrain_red[:, :, z].T, cmap='Reds')
    ax[z_ind, 1].imshow(template_transformed[:, :, z].T, cmap='Blues')
    ax[z_ind, 2].imshow(glom_tmp[:, :, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind==0:
        ax[z_ind, 0].set_title('mtdTomato meanbrain')
        ax[z_ind, 1].set_title('JRC2018')
        ax[z_ind, 2].set_title('Glomerulus map')

        dx = 25 / 1.0 # um -> pix
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
