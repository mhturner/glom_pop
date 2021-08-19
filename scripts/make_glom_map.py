"""
Process JRC2018 template brain & glomerulus map

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import os
import numpy as np
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from skimage import morphology
import colorcet as cc
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import ants

from glom_pop import dataio

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'

# %% LOAD TEMPLATE ARRAYS
# (1) Load
# (2) Get in xyz order
# (3) Trim down to left PVLP/PLP area: [410:645, 250:450, 230:340] xyz
# (4) Flip along z axis. Template has anterior at the top, but want anterior at the bottom of the stack

# Load glom map hdf5 as array
fileh = h5py.File(os.path.join(base_dir, 'template_brain', 'vpn_glom_map.h5'), 'r')  # dim order = zxy

# Mask with VPN identity
brain_mask = np.zeros(fileh.get('mask/array').shape, dtype='uint8')
fileh['mask/array'].read_direct(brain_mask)
brain_mask = np.moveaxis(brain_mask, (0, 1, 2), (2, 0, 1))  # to xyz

# Density map
brain_density = np.zeros(fileh.get('density/array').shape, dtype='uint8')
fileh['density/array'].read_direct(brain_density)
brain_density = np.moveaxis(brain_density, (0, 1, 2), (2, 0, 1))  # to xyz

fileh.close()

print('Full brain_mask shape = {}'.format(brain_mask.shape))
brain_mask = np.flip(brain_mask[410:645, 250:450, 230:340], axis=2)
brain_density = np.flip(brain_density[410:645, 250:450, 230:340], axis=2)
print('Trimmed brain_mask shape = {}'.format(brain_mask.shape))

# Load template brain
template = np.squeeze(np.asanyarray(nib.load(os.path.join(base_dir, 'template_brain', 'JRC2018_FEMALE_38um_iso_16bit.nii')).dataobj).astype('uint32')) # xyz
print('Full template shape = {}'.format(template.shape))
template = np.flip(template[410:645, 250:450, 230:340], axis=2)
print('Trimmed template shape = {}'.format(template.shape))

# Save trimmed template & mask (pre-transformation)
nib.save(nib.Nifti1Image(template, np.eye(4)), os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask.nii'))
nib.save(nib.Nifti1Image(brain_density, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_density.nii'))
# %% Register density -> template

# Convert to ANTS images
atlas_spacing = (0.38, 0.38, 0.38)  # um

template = ants.from_numpy(template, spacing=atlas_spacing)
brain_density = ants.from_numpy(brain_density, spacing=atlas_spacing)
brain_mask = ants.from_numpy(brain_mask, spacing=atlas_spacing)

# registration images: Smoothed template and closed density map
template_smoothed = dataio.get_smooth_brain(template, smoothing_sigma=[3, 3, 2])
brain_density_closed = morphology.closing(brain_density.numpy() > 0, selem=morphology.ball(4))
brain_density_closed = brain_density_closed * 1.0  # bool -> float
brain_density_closed = ants.from_numpy(brain_density_closed, spacing=atlas_spacing)

# %% Register density -> template

reg = ants.registration(template_smoothed,  # fixed = template, nc82, smoothed
                        brain_density_closed,  # Moving = syn density, closed
                        type_of_transform='SyN',
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=1)

# Apply alignment to brain density and mask
brain_density_transformed = ants.apply_transforms(template,
                                                  brain_density,
                                                  reg['fwdtransforms'],
                                                  interpolator='nearestNeighbor')

brain_mask_transformed = ants.apply_transforms(template,
                                               brain_mask,
                                               reg['fwdtransforms'],
                                               interpolator='genericLabel')

# Save transformed mask & density
nib.save(nib.Nifti1Image(brain_mask_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_transformed.nii'))
nib.save(nib.Nifti1Image(brain_density_transformed.numpy(), np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_density_transformed.nii'))

# %% MORPHOLOGICAL OPERATIONS ON EACH GLOMERULUS MASK

# Back to numpy arrays, don't need ants anymore
brain_mask_transformed = brain_mask_transformed.numpy()

mask_ids = np.unique(brain_mask_transformed)[1:]  # exclude first (=0, i.e. nothing)

closed_eroded_mask = np.zeros_like(brain_mask_transformed)
for mask_id in mask_ids:
    # Closing
    morph_mask = morphology.closing(brain_mask_transformed == mask_id, selem=morphology.ball(4))
    # One more erosion to remove speckles
    morph_mask = morphology.binary_erosion(morph_mask, selem=morphology.ball(1))
    closed_eroded_mask[morph_mask] = mask_id

# Save closed mask
nib.save(nib.Nifti1Image(closed_eroded_mask, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))

# %% SHOW
z_slice = 60

norm = mcolors.Normalize(vmin=1, vmax=brain_mask_transformed.max(), clip=True)

fh, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(template[:, :, z_slice].T, cmap='Blues')
ax[0].set_title('JRC2018')

ax[1].imshow(np.ma.masked_where(brain_mask_transformed == 0, brain_mask_transformed)[:, :, z_slice].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[1].set_title('Density Map')

ax[2].imshow(np.ma.masked_where(closed_eroded_mask == 0, closed_eroded_mask)[:, :, z_slice].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[2].set_title('Closed Map')


for x in ax.ravel():
    x.locator_params(axis='y', nbins=6)
    x.locator_params(axis='x', nbins=10)
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.grid(which='minor', axis='both', linestyle='--', color='k')
