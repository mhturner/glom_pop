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
# meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')

mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format('20210824')
# %% LOAD
# Load meanbrain
meanbrain = ants.image_read(os.path.join(base_dir, 'mean_brain', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brain', mask_fn)).dataobj).astype('uint32')
lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain_red.spacing)

# Load template & glom mask
template = ants.image_read(os.path.join(base_dir, 'template_brain', 'jrc2018.nii'))
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %% SHOW TEMPLATE BRAIN
template_smoothed = ants.smooth_image(template, sigma=[3, 3, 2], sigma_in_physical_coordinates=False)
# trim Z a bit to exclude very deep stuff not included in meanbrain
template_smoothed = ants.from_numpy(template_smoothed.numpy()[:, :, 0:70], spacing=template.spacing)

fh, ax = plt.subplots(3, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(12):
    ax[z].imshow(template_smoothed[:, :, z*6].T)

# %% SHOW MEAN BRAIN - GREEN. Align on this.

meanbrain_smoothed = ants.smooth_image(meanbrain_green, sigma=[2, 2, 1], sigma_in_physical_coordinates=False)

disp_masked = np.ma.masked_where(lobe_mask.numpy() == 0, meanbrain_smoothed.numpy())  # mask at 0

fh, ax = plt.subplots(5, 5, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(23):
    ax[z].imshow(disp_masked[:, :, z*2].T)
# %% COMPUTE AFFINE ALIGNMENT

fixed_im = ants.n4_bias_field_correction(meanbrain_smoothed)
moving_im = ants.n4_bias_field_correction(template_smoothed)

reg_aff = ants.registration(meanbrain_smoothed,  # fixed = meanbrain, green channel
                            template_smoothed,  # Moving = smoothed template
                            type_of_transform='Affine',
                            mask=lobe_mask,  # mask includes only gloms, excludes lobe
                            flow_sigma=6,
                            total_sigma=0,
                            random_seed=1)

transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template', 'affine')
os.makedirs(transform_dir, exist_ok=True)

# APPLY ALIGNMENT TO MASK & TEMPLATE
template_aff = ants.apply_transforms(fixed=meanbrain_green,
                                     moving=template,
                                     transformlist=reg_aff['fwdtransforms'],
                                     interpolator='nearestNeighbor')

glom_mask_aff = ants.apply_transforms(fixed=meanbrain_green,
                                      moving=glom_mask,
                                      transformlist=reg_aff['fwdtransforms'],
                                      interpolator='genericLabel')

ants.image_write(template_aff, os.path.join(transform_dir, 'JRC2018_affine2meanbrain.nii'))
ants.image_write(glom_mask_aff, os.path.join(transform_dir, 'glom_mask_affine2meanbrain.nii'))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_aff, glom_mask_aff])
save_path = os.path.join(transform_dir, 'overlay_meanbrain_affine.nii')
ants.image_write(merged, save_path)

# Then compute a small warp on top of this
type_of_transform = 'SyN'
reg_warp = ants.registration(meanbrain_green,  # fixed = meanbrain, green channel
                             template_aff,  # Moving = template
                             type_of_transform=type_of_transform,
                             mask=lobe_mask,  # mask includes only gloms, excludes lobe
                             flow_sigma=1,
                             total_sigma=20,
                             random_seed=1)

# # Save transform
# transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template_affine')
# os.makedirs(transform_dir, exist_ok=True)
# dataio.save_transforms(reg, transform_dir)
#
# # Apply alignment to brain density and mask
# transform_list = dataio.get_transform_list(transform_dir, direction='forward')

# APPLY ALIGNMENT TO MASK & TEMPLATE
template_warped = ants.apply_transforms(fixed=meanbrain_green,
                                        moving=template_aff,
                                        transformlist=reg_warp['fwdtransforms'],
                                        interpolator='nearestNeighbor')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_green,
                                         moving=glom_mask_aff,
                                         transformlist=reg_warp['fwdtransforms'],
                                         interpolator='genericLabel')

ants.image_write(template_warped, os.path.join(transform_dir, 'JRC2018_affine_{}_2meanbrain.nii'.format(type_of_transform)))
ants.image_write(glom_mask_warped, os.path.join(transform_dir, 'glom_mask_affine_{}_2meanbrain.nii'.format(type_of_transform)))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, 'overlay_meanbrain_affine_{}_.nii'.format(type_of_transform))
ants.image_write(merged, save_path)

# %% CHECK OVERALL ALIGNMENT
fh, ax = plt.subplots(3, 3, figsize=(12, 9))
# [util.cleanAxes(x) for x in ax.ravel()]

ax[0, 1].imshow(meanbrain_green.max(axis=2).T, vmax=np.quantile(meanbrain_green.max(axis=2).ravel(), 0.95))
ax[0, 1].grid(which='major', axis='both', linestyle='-', color='r')

ax[1, 0].imshow(template.max(axis=2).T)
ax[1, 1].set_title('Template')

ax[1, 1].imshow(template_aff.max(axis=2).T)
ax[1, 1].set_title('Affine')
ax[1, 1].grid(which='major', axis='both', linestyle='-', color='r')

ax[1, 2].imshow(template_warped.max(axis=2).T)
ax[1, 2].set_title('+warp')
ax[1, 2].grid(which='major', axis='both', linestyle='-', color='r')

warp_grid = create_warped_grid(template,
                               grid_directions=(True, True), grid_step=20, grid_width=2,
                               transform=None, fixed_reference_image=meanbrain_green)
ax[2, 0].imshow(warp_grid.mean(axis=2).T, cmap='Greys_r')


warp_grid = create_warped_grid(template,
                               grid_directions=(True, True), grid_step=20, grid_width=2,
                               transform=reg_aff['fwdtransforms'], fixed_reference_image=meanbrain_green)
ax[2, 1].imshow(warp_grid.mean(axis=2).T, cmap='Greys_r')

warp_grid = create_warped_grid(template_aff,
                               grid_directions=(True, True), grid_step=10, grid_width=1,
                               transform=reg_warp['fwdtransforms'], fixed_reference_image=meanbrain_green)
ax[2, 2].imshow(warp_grid[:, :, 20].T, cmap='Greys_r')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# %% COMPUTE ALIGNMENT - SyN

reg = ants.registration(meanbrain_smoothed,  # fixed = meanbrain, green channel
                        template_smoothed,  # Moving = smoothed template
                        type_of_transform='SyN',
                        mask=lobe_mask,  # mask includes only gloms, excludes lobe
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=1,
                        reg_iterations=[200, 50, 20])

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
ax[0].imshow(meanbrain_green.max(axis=2).T, vmax=np.quantile(meanbrain_green.max(axis=2).ravel(), 0.9))
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

# %% Playing with reg

# %% COMPUTE ALIGNMENT IN TWO STAGES
# 1) AFFINE
# 2) SyN
# Breaking it up helps troubleshoot. Make sure affine is close first, then mess with SyN warp if necessary

# (1) Affine alignment
reg_aff = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_green),
                            moving=ants.n4_bias_field_correction(template),
                            type_of_transform='Affine',
                            mask=lobe_mask,  # mask includes only gloms, excludes lobe
                            flow_sigma=6,
                            total_sigma=0,
                            random_seed=0)

# (2) Then compute a small SyN warp on top of this. MI metric.
# output reg_syn transform contains both this new syn warp and the initial affine transform
reg_syn = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_green),
                            moving=ants.n4_bias_field_correction(template),
                            type_of_transform='SyNOnly',
                            mask=lobe_mask,  # mask includes only gloms, excludes lobe
                            initial_transform=reg_aff['fwdtransforms'][0],
                            random_seed=0)

# # Concatenate transforms to make final transform list. Note order here is warp then affine
# transform_list = [
#                   reg_syn['fwdtransforms'][0],  # Warp
#                   reg_aff['fwdtransforms'][0]   # Affine
#                   ]
# APPLY ALIGNMENT TO MASK & TEMPLATE
template_warped = ants.apply_transforms(fixed=meanbrain_green,
                                        moving=template,
                                        transformlist=reg_syn['fwdtransforms'],
                                        interpolator='nearestNeighbor')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_green,
                                         moving=glom_mask,
                                         transformlist=reg_syn['fwdtransforms'],
                                         interpolator='genericLabel')

# Save transform
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')
os.makedirs(transform_dir, exist_ok=True)
dataio.save_transforms(reg_syn, transform_dir)

# Save transformed images
ants.image_write(template_warped, os.path.join(transform_dir, 'JRC2018_reg2meanbrain.nii'))
ants.image_write(glom_mask_warped, os.path.join(transform_dir, 'glom_mask_reg2meanbrain.nii'))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, 'overlay_meanbrain.nii')
ants.image_write(merged, save_path)

# %% CHECK OVERALL ALIGNMENT
mi_orig = ants.image_mutual_information(meanbrain_green, template)
mi_aff = ants.image_mutual_information(meanbrain_green, reg_aff['warpedmovout'])
mi_syn = ants.image_mutual_information(meanbrain_green, template_warped)

fh, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(meanbrain_green.max(axis=2).T, vmax=np.quantile(meanbrain_green.max(axis=2).ravel(), 0.9))
ax[0].set_title('Original, MI = {:.2f}'.format(mi_orig))

ax[1].imshow(reg_aff['warpedmovout'].max(axis=2).T)
ax[1].set_title('Affine, MI = {:.2f}'.format(mi_aff))
ax[2].imshow(template_warped.max(axis=2).T)
ax[2].set_title('+ SyN Warp, MI = {:.2f}'.format(mi_syn))

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='-', color='r')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# %%
