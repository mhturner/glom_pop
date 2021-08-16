"""
Make Anatomical Meanbrain

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import os
import numpy as np
import ants
import nibabel as nib
import matplotlib.pyplot as plt
import time
import datetime

from glom_pop import dataio

# %% ANATOMICAL SCAN FILES

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'

# file_names = ['ZSeries-20210804-001',  # **reference brain**
#               'ZSeries-20210804-004',  #
#               'ZSeries-20210804-007',  #
#               'ZSeries-20210811-001',  #
#               'ZSeries-20210811-004',  #
#               'ZSeries-20210811-007',  #
#               ]

file_names = [
              # 'TSeries-20210804-003',  # **reference brain**
              # 'TSeries-20210804-006',  #
              # 'TSeries-20210804-009',  #
              'TSeries-20210811-003',  #
              'TSeries-20210811-006',  #
              'TSeries-20210811-009',  #
              ]


today = datetime.datetime.today().strftime('%Y%m%d')

# %% REFERENCE BRAIN
reference_index = 0
reference_fn = file_names[reference_index]

filepath = os.path.join(base_dir, 'anatomical_brains', reference_fn)
metadata = dataio.get_bruker_metadata(filepath + '.xml')

reference = dataio.get_ants_brain(filepath + '_channel_1.nii', metadata)  # xyz
reference = ants.n3_bias_field_correction(reference)
spacing = reference.spacing

print('Brain spacing is {}'.format(spacing))

plt.imshow(reference.max(axis=2).T)

# %% MAKE MEANBRAIN_1:
# Register each brain to reference brain
# Meanbrain 1: register each to reference (smoothed)
t0 = time.time()
corrected_brains = []
for fn in file_names:
    filepath = os.path.join(base_dir, 'anatomical_brains', fn)
    metadata = dataio.get_bruker_metadata(filepath + '.xml')

    red_brain = dataio.get_ants_brain(filepath + '_channel_1.nii', metadata)  # xyz
    # bias correction
    red_brain = ants.n3_bias_field_correction(red_brain)

    reg = ants.registration(dataio.get_smooth_brain(reference, smoothing_sigma=[1.0, 1.0, 0.0]),
                            dataio.get_smooth_brain(red_brain, smoothing_sigma=[1.0, 1.0, 0.0]),
                            type_of_transform='SyN',
                            flow_sigma=3,
                            total_sigma=0)

    red_reg = ants.apply_transforms(fixed=reference,
                                    moving=red_brain,
                                    transformlist=reg['fwdtransforms'],
                                    interpolator='nearestNeighbor',
                                    defaultvalue=0)

    red_reg = red_reg.numpy().astype('float')
    red_reg[red_reg == 0] = np.nan

    corrected_brains.append(red_reg)

meanbrain_1 = np.nanmean(np.stack(corrected_brains, -1), axis=-1)
meanbrain_1[np.isnan(meanbrain_1)] = 0
meanbrain_1 = ants.from_numpy(meanbrain_1, spacing=spacing)
print('Computed meanbrain 1 ({} sec)'.format(time.time()-t0))

# %% CHECK INITIAL CORRECTED BRAINS
# Should roughly align. Spot any problem brains...

stride = 5  # show every n z slices

fh, ax = plt.subplots(1, len(file_names), figsize=(12, 4))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
for b_ind, b in enumerate(corrected_brains):
    ax[b_ind].imshow(np.nanmean(b, axis=2).T, cmap='Reds')

# Show z slices of meanbrain1 vs reference brain
fh, ax = plt.subplots(len(file_names)+1, int(meanbrain_1.shape[2]/stride), figsize=(24, 2*len(file_names)))
[x. set_axis_off() for x in ax.ravel()]
for z in range(int(meanbrain_1.shape[2]/stride)):
    ax[0, z].imshow(meanbrain_1[:, :, z*stride].T)
    for cb_ind, cb in enumerate(corrected_brains):
        ax[cb_ind+1, z].imshow(cb[:, :, z*stride].T)
    if z == 0:
        ax[0, z].set_title('meanbrain')
        ax[reference_index+1, z].set_title('reference')


# %% FINAL MEANBRAIN:
# Register each brain to meanbrain_1, save meanbrain

t0 = time.time()
corrected_red = []
corrected_green = []
for fn in file_names:
    date_str = fn.split('-')[1]
    filepath = os.path.join(base_dir, 'anatomical_brains', fn)
    metadata = dataio.get_bruker_metadata(filepath + '.xml')

    red_brain = dataio.get_ants_brain(filepath + '_channel_1.nii', metadata)  # xyz, red
    green_brain = dataio.get_ants_brain(filepath + '_channel_2.nii', metadata)  # xyz, green

    reg = ants.registration(meanbrain_1,
                            ants.n3_bias_field_correction(red_brain),  # Note bias correction
                            type_of_transform='ElasticSyN',  # Note Elastic transformation now!
                            flow_sigma=3,
                            total_sigma=0)

    red_reg = ants.apply_transforms(fixed=reference,
                                    moving=red_brain,
                                    transformlist=reg['fwdtransforms'],
                                    interpolator='nearestNeighbor',
                                    defaultvalue=0)

    green_reg = ants.apply_transforms(fixed=reference,
                                      moving=green_brain,
                                      transformlist=reg['fwdtransforms'],
                                      interpolator='nearestNeighbor',
                                      defaultvalue=0)

    red_reg = red_reg.numpy().astype('float')
    red_reg[red_reg == 0] = np.nan

    green_reg = green_reg.numpy().astype('float')
    green_reg[green_reg == 0] = np.nan

    corrected_red.append(red_reg)
    corrected_green.append(green_reg)

meanbrain_red = np.nanmean(np.stack(corrected_red, -1), axis=-1)
meanbrain_green = np.nanmean(np.stack(corrected_green, -1), axis=-1)

# occluded back to 0
meanbrain_red[np.isnan(meanbrain_red)] = 0
meanbrain_green[np.isnan(meanbrain_green)] = 0

merged = dataio.merge_channels(meanbrain_red, meanbrain_green)
nib.save(nib.Nifti1Image(merged.astype('uint16'), np.eye(4)), os.path.join(base_dir, 'mean_brains', 'chat_meanbrain_{}.nii'.format(today)))

# Save single channels
nib.save(nib.Nifti1Image(meanbrain_red.astype('uint16'), np.eye(4)), os.path.join(base_dir, 'mean_brains', 'chat_meanbrain_{}_ch1.nii'.format(today)))
nib.save(nib.Nifti1Image(meanbrain_green.astype('uint16'), np.eye(4)), os.path.join(base_dir, 'mean_brains', 'chat_meanbrain_{}_ch2.nii'.format(today)))

print('Computed and saved final meanbrain ({} sec)'.format(time.time()-t0))

# Convert red to ants for next registration step
meanbrain_red = ants.from_numpy(meanbrain_red, spacing=spacing)
# %% REGISTER EACH BRAIN TO MEANBRAIN

for f_ind, fn in enumerate(file_names):
    # # # Compute transform and load raw brain channels # # #
    t0 = time.time()

    filepath = os.path.join(base_dir, 'anatomical_brains', fn)
    metadata = dataio.get_bruker_metadata(filepath + '.xml')

    red_brain = dataio.get_ants_brain(filepath + '_channel_1.nii', metadata)  # xyz
    green_brain = dataio.get_ants_brain(filepath + '_channel_2.nii', metadata)  # xyz

    reg = ants.registration(meanbrain_red,
                            red_brain,
                            type_of_transform='SyN',
                            flow_sigma=3,
                            total_sigma=0)

    print('Computed transform and loaded brains: {} ({} sec)'.format(fn, time.time()-t0))

    # # # Apply transform to each channel # # #
    t0 = time.time()
    red_reg = ants.apply_transforms(fixed=meanbrain_red,
                                    moving=red_brain,
                                    transformlist=reg['fwdtransforms'],
                                    interpolator='nearestNeighbor',
                                    defaultvalue=0)
    green_reg = ants.apply_transforms(fixed=meanbrain_red,
                                      moving=green_brain,
                                      transformlist=reg['fwdtransforms'],
                                      interpolator='nearestNeighbor',
                                      defaultvalue=0)
    print('Applied transforms to {} ({} sec)'.format(fn, time.time()-t0))
    del red_brain, green_brain

    # # # Save # # #
    # Save meanbrain as .nii
    merged = dataio.merge_channels(red_reg.numpy(), green_reg.numpy())
    nib.save(nib.Nifti1Image(merged.astype('uint16'), np.eye(4)), os.path.join(base_dir, 'mean_brains', fn) + '_meanbrain.nii')
    print('Saved to {}'.format(os.path.join(base_dir, 'mean_brains', fn) + '_meanbrain.nii'))

# %%
