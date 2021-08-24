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
from skimage import morphology
import colorcet as cc
import matplotlib.colors as mcolors
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
template = np.squeeze(np.asanyarray(nib.load(os.path.join(base_dir, 'template_brain', 'JRC2018_FEMALE_38um_iso_16bit.nii')).dataobj).astype('uint32'))  # xyz
print('Full template shape = {}'.format(template.shape))
template = np.flip(template[410:645, 250:450, 230:340], axis=2)
print('Trimmed template shape = {}'.format(template.shape))

# Convert to ANTS images
atlas_spacing = (0.38, 0.38, 0.38)  # um
template = ants.from_numpy(template, spacing=atlas_spacing)
brain_mask = ants.from_numpy(brain_mask, spacing=atlas_spacing)
brain_density = ants.from_numpy(brain_density, spacing=atlas_spacing)

# Save trimmed template & mask (pre-transformation) as ants images
ants.image_write(template, os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
ants.image_write(brain_mask, os.path.join(base_dir, 'template_brain', 'vpn_glom_mask.nii'))
ants.image_write(brain_density, os.path.join(base_dir, 'template_brain', 'vpn_glom_density.nii'))

# %% Register density -> template

# Load:
template = ants.image_read(os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
brain_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask.nii'))
brain_density = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_density.nii'))

# registration images: Smoothed template and closed density map
template_smoothed = ants.smooth_image(template, sigma=[3, 3, 2], sigma_in_physical_coordinates=False)
brain_density_closed = morphology.closing(brain_density.numpy() > 0, selem=morphology.ball(4))
brain_density_closed = brain_density_closed * 1.0  # bool -> float
brain_density_closed = ants.from_numpy(brain_density_closed, spacing=atlas_spacing)

# Compute registration (density -> template)
reg = ants.registration(fixed=template_smoothed,  # fixed = template, nc82, smoothed
                        moving=brain_density_closed,  # Moving = syn density, closed
                        type_of_transform='SyN',
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=1)

# %%
# Save transform
transform_dir = os.path.join(base_dir, 'transforms', 'template_density')
os.makedirs(transform_dir, exist_ok=True)
dataio.save_transforms(reg, transform_dir)

# Apply alignment to brain density and mask
transform_list = dataio.get_transform_list(transform_dir, direction='forward')
brain_density_transformed = ants.apply_transforms(fixed=template,
                                                  moving=brain_density,
                                                  transformlist=transform_list,
                                                  interpolator='nearestNeighbor')

brain_mask_transformed = ants.apply_transforms(fixed=template,
                                               moving=brain_mask,
                                               transformlist=transform_list,
                                               interpolator='genericLabel')

# Save transformed mask & density
ants.image_write(brain_mask_transformed, os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_transformed.nii'))
ants.image_write(brain_density_transformed, os.path.join(base_dir, 'template_brain', 'vpn_glom_density_transformed.nii'))

# %% MORPHOLOGICAL OPERATIONS ON EACH GLOMERULUS MASK

# Load:
brain_mask_transformed = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_transformed.nii'))

mask_ids = np.unique(brain_mask_transformed.numpy())[1:]  # exclude first (=0, i.e. nothing)

closed_eroded_mask = np.zeros_like(brain_mask_transformed.numpy())
for mask_id in mask_ids:
    # Closing
    morph_mask = morphology.closing(brain_mask_transformed.numpy() == mask_id, selem=morphology.ball(4))
    # One more erosion to remove speckles
    morph_mask = morphology.binary_erosion(morph_mask, selem=morphology.ball(1))
    closed_eroded_mask[morph_mask] = mask_id

# Convert to ANTs image & save: closed mask
closed_eroded_mask = ants.from_numpy(closed_eroded_mask, spacing=brain_mask_transformed.spacing)
ants.image_write(closed_eroded_mask, os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))

# %% SHOW
z_slice = 60

# Load:
brain_density_transformed = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_density_transformed.nii'))
closed_eroded_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))
template = ants.image_read(os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))

norm = mcolors.Normalize(vmin=1, vmax=closed_eroded_mask.max(), clip=True)

fh, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(template[:, :, z_slice].T, cmap='Blues')
ax[0].set_title('JRC2018')

ax[1].imshow(np.ma.masked_where(brain_density_transformed.numpy() == 0, brain_density_transformed.numpy())[:, :, z_slice].T,
                                norm=norm, interpolation='none')
ax[1].set_title('Density Map')

ax[2].imshow(np.ma.masked_where(closed_eroded_mask.numpy() == 0, closed_eroded_mask.numpy())[:, :, z_slice].T,
                                cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[2].set_title('Closed Map')


for x in ax.ravel():
    x.locator_params(axis='y', nbins=6)
    x.locator_params(axis='x', nbins=10)
    x.grid(which='major', axis='both', linestyle='--', color='k')
    x.grid(which='minor', axis='both', linestyle='--', color='k')
