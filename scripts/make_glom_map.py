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


# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# Save trimmed template & raw mask
nib.save(nib.Nifti1Image(template, np.eye(4)), os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask.nii'))
nib.save(nib.Nifti1Image(brain_density, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_density.nii'))


# %% MORPHOLOGICAL OPERATIONS ON EACH GLOMERULUS MASK

mask_ids = np.unique(brain_mask)[1:]  # exclude first (=0, i.e. nothing)

closed_mask = np.zeros_like(brain_mask)
closed_eroded_mask = np.zeros_like(brain_mask)
for mask_id in mask_ids:
    morph_mask = morphology.closing(brain_mask == mask_id, selem=morphology.ball(4))
    closed_mask[morph_mask] = mask_id

    morph_mask = morphology.binary_erosion(morph_mask, selem=morphology.ball(1))
    closed_eroded_mask[morph_mask] = mask_id

# Save closed masks
nib.save(nib.Nifti1Image(closed_mask, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))  # morpho closed each glom
nib.save(nib.Nifti1Image(closed_eroded_mask, np.eye(4)), os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed_eroded.nii'))  # morpho closed each glom

# %% SHOW
z_slice = 60

norm = mcolors.Normalize(vmin=1, vmax=brain_mask.max(), clip=True)

fh, ax = plt.subplots(1, 4, figsize=(18, 6))
ax[0].imshow(template[:, :, z_slice].T, cmap='Blues')
ax[0].set_title('JRC2018')

ax[1].imshow(np.ma.masked_where(brain_mask == 0, brain_mask)[:, :, z_slice].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[1].set_title('Raw Density Map')

ax[2].imshow(np.ma.masked_where(closed_mask == 0, closed_mask)[:, :, z_slice].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[2].set_title('Closed Map')

ax[3].imshow(np.ma.masked_where(closed_eroded_mask == 0, closed_eroded_mask)[:, :, z_slice].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[3].set_title('Closed, Eroded Map')


for x in ax.ravel():
    x.locator_params(axis='y', nbins=6)
    x.locator_params(axis='x', nbins=10)
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.grid(which='minor', axis='both', linestyle='--', color='k')
