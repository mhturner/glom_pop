"""
Make Meanbrain from anatomical scans

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import os
import numpy as np
import ants
import matplotlib.pyplot as plt
import time
import datetime
import glob
import nibabel as nib

from glom_pop import dataio

# %% ANATOMICAL SCAN FILES

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
today = datetime.datetime.today().strftime('%Y%m%d')

# %% REFERENCE BRAIN
reference_filename = 'TSeries-20210804-009_anatomical.nii'
# 2-channel xyz
reference_brain = ants.image_read(os.path.join(base_dir, 'mean_brain', reference_filename))
spacing = reference_brain.spacing

print('Brain spacing is {}'.format(spacing))

fh, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(ants.split_channels(reference_brain)[0].max(axis=2).T, cmap='Reds')
ax[1].imshow(ants.split_channels(reference_brain)[1].max(axis=2).T, cmap='Greens')

# %%


def registerBrainsToReference(brain_file_path,
                              reference_brain,
                              type_of_transform='ElasticSyN',
                              flow_sigma=3,
                              total_sigma=0,
                              initial_transform=None,
                              mask=None,
                              do_bias_correction=False):
    """
    Register each brain in brain_directory to reference_brain.
        Saves registered brain image and transform file

    :brain_file_path: .nii filepath to brain to register (MOVING)
    :reference_brain: two-channel ANTs image to register each brain to (FIXED)
    :type_of_transform: for ants.registration()
    """

    print('Starting brain {}'.format(brain_file_path))
    t0 = time.time()
    series_name = os.path.split(brain_file_path)[-1].split('_')[0]
    base_dir = os.path.split(os.path.split(brain_file_path)[0])[0]

    # Make save paths for transforms
    transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_anatomical', series_name)
    os.makedirs(transform_dir, exist_ok=True)
    os.makedirs(os.path.join(transform_dir, 'forward'), exist_ok=True)
    os.makedirs(os.path.join(transform_dir, 'inverse'), exist_ok=True)

    individual_brain = ants.image_read(brain_file_path)

    if do_bias_correction:
        fixed_red = ants.n4_bias_field_correction(ants.split_channels(reference_brain)[0])
        moving_red = ants.n4_bias_field_correction(ants.split_channels(individual_brain)[0])

    else:
        fixed_red = ants.split_channels(reference_brain)[0]
        moving_red = ants.split_channels(individual_brain)[0]

    reg = ants.registration(fixed=fixed_red,
                            moving=moving_red,
                            type_of_transform=type_of_transform,
                            flow_sigma=flow_sigma,
                            total_sigma=total_sigma,
                            mask=mask,
                            initial_transform=initial_transform)

    # Copy transforms from tmp to long-term save dir
    dataio.save_transforms(reg, transform_dir)

    # Apply transform to each channel
    transform_list = dataio.get_transform_list(transform_dir, direction='forward')

    red_reg = ants.apply_transforms(fixed=ants.split_channels(reference_brain)[0],
                                    moving=ants.split_channels(individual_brain)[0],
                                    transformlist=transform_list,
                                    interpolator='nearestNeighbor',
                                    defaultvalue=0)

    green_reg = ants.apply_transforms(fixed=ants.split_channels(reference_brain)[1],
                                      moving=ants.split_channels(individual_brain)[1],
                                      transformlist=transform_list,
                                      interpolator='nearestNeighbor',
                                      defaultvalue=0)

    # Merge channels and save registered anatomical scan
    merged = ants.merge_channels([red_reg, green_reg])
    save_path = os.path.join(transform_dir,  'meanbrain_reg.nii')
    ants.image_write(merged, save_path)

    # Merge and save an overlay to check the registration
    overlay = ants.merge_channels([ants.split_channels(reference_brain)[0], red_reg])
    save_path = os.path.join(transform_dir,  'overlay_reg.nii')
    ants.image_write(overlay, save_path)

    print('Computed and saved transforms to {} ({} sec)'.format(transform_dir, time.time()-t0))
    print('--------------------')


def computeMeanbrain(brain_directory,
                     reference_brain,
                     type_of_transform='SyN',
                     do_bias_correction=True,
                     do_smoothing=True):
    """
    Generate a meanbrain from a list of anatomical scans.

    :brain_directory: contains anatomical ANTs images to register, fns end in '_anatomical.nii'
    :reference_brain: two-channel ANTs image to register each brain to
    :type_of_transform: for ants.registration()
    :do_bias_correction: [bool]
    :do_smoothing: [bool]
    """
    t0 = time.time()
    corrected_red = []
    corrected_green = []
    file_paths = glob.glob(os.path.join(brain_directory, '*_anatomical.nii'))
    for fp in file_paths:
        t0_fp = time.time()
        individual_red = ants.split_channels(ants.image_read(fp))[0]
        individual_green = ants.split_channels(ants.image_read(fp))[1]

        # Temporary copy to use to compute registration
        moving_red = individual_red.clone()
        fixed_red = ants.split_channels(reference_brain)[0]
        if do_bias_correction:
            moving_red = ants.n3_bias_field_correction(moving_red)
        if do_smoothing:
            fixed_red = ants.smooth_image(fixed_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)
            moving_red = ants.smooth_image(moving_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)

        reg = ants.registration(fixed=fixed_red,
                                moving=moving_red,
                                type_of_transform=type_of_transform,
                                flow_sigma=3,
                                total_sigma=0)

        red_reg = ants.apply_transforms(fixed=reference_brain,
                                        moving=individual_red,
                                        transformlist=reg['fwdtransforms'],
                                        interpolator='nearestNeighbor',
                                        defaultvalue=0)

        green_reg = ants.apply_transforms(fixed=reference_brain,
                                          moving=individual_green,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='nearestNeighbor',
                                          defaultvalue=0)

        red_reg = red_reg.numpy().astype('float')
        red_reg[red_reg == 0] = np.nan

        green_reg = green_reg.numpy().astype('float')
        green_reg[green_reg == 0] = np.nan

        corrected_red.append(red_reg)
        corrected_green.append(green_reg)

        print('Done with brain {} ({:.2f} sec)'.format(fp.split('/')[-1], time.time()-t0_fp))

    meanbrain_red = np.nanmean(np.stack(corrected_red, -1), axis=-1)
    meanbrain_green = np.nanmean(np.stack(corrected_green, -1), axis=-1)

    # occluded back to 0
    meanbrain_red[np.isnan(meanbrain_red)] = 0
    meanbrain_green[np.isnan(meanbrain_green)] = 0

    # Convert to ANTs image and merge channels
    meanbrain_red = ants.from_numpy(meanbrain_red, spacing=reference_brain.spacing)
    meanbrain_green = ants.from_numpy(meanbrain_green, spacing=reference_brain.spacing)
    meanbrain = ants.merge_channels([meanbrain_red, meanbrain_green])

    print('Computed meanbrain ({} sec)'.format(time.time()-t0))
    return meanbrain


def showBrain(brain, stride):
    """Quick display z slices of 2 channel brain."""
    num_slices = int(brain.shape[2]/stride)
    fh, ax = plt.subplots(2, num_slices, figsize=(num_slices*3, 4))
    [x.set_axis_off() for x in ax.ravel()]
    for z_ind in range(num_slices):
        ax[0, z_ind].imshow(ants.split_channels(brain)[0][:, :, z_ind*stride].T, cmap='Reds')
        ax[1, z_ind].imshow(ants.split_channels(brain)[1][:, :, z_ind*stride].T, cmap='Greens')


# %% Compute meanbrain 1:
# Align smoothed brains, to get things roughly aligned
meanbrain_1 = computeMeanbrain(brain_directory=os.path.join(base_dir, 'mean_brain'),
                               reference_brain=reference_brain,
                               type_of_transform='SyN',
                               do_bias_correction=True,
                               do_smoothing=True)

showBrain(meanbrain_1, stride=8)

# %% Compute meanbrain 2:
# No smoothing, elastic alignment
meanbrain_2 = computeMeanbrain(brain_directory=os.path.join(base_dir, 'mean_brain'),
                               reference_brain=meanbrain_1,
                               type_of_transform='ElasticSyN',
                               do_bias_correction=True,
                               do_smoothing=False)

showBrain(meanbrain_2, stride=8)
# %% Compute final meanbrain:
meanbrain = computeMeanbrain(brain_directory=os.path.join(base_dir, 'mean_brain'),
                             reference_brain=meanbrain_2,
                             type_of_transform='ElasticSyN',
                             do_bias_correction=True,
                             do_smoothing=False)

showBrain(meanbrain, stride=8)

# Save final meanbrain
save_path = os.path.join(base_dir, 'mean_brain', 'chat_meanbrain_{}.nii'.format(today))
ants.image_write(meanbrain, save_path)

# save individual channels
ants.image_write(ants.split_channels(meanbrain)[0], os.path.join(base_dir, 'mean_brain', 'chat_meanbrain_{}_ch1.nii'.format(today)))
ants.image_write(ants.split_channels(meanbrain)[1], os.path.join(base_dir, 'mean_brain', 'chat_meanbrain_{}_ch2.nii'.format(today)))


# %% Register each brain to final meanbrain and save these transforms

# Load meanbrain
# meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')
meanbrain = ants.image_read(os.path.join(base_dir, 'mean_brain', meanbrain_fn))

# Note dir. change: registering all anatomical brains to meanbrain now
file_paths = glob.glob(os.path.join(base_dir, 'anatomical_brains', '*_anatomical.nii'))

# %% Run through all brains in dir
# Register each anatomical brain to the meanbrain
# For each anatomical brain, saves:
#   -forward (anat -> meanbrain) & inverse (meanbrain -> anat) transforms

for brain_file_path in file_paths:
    registerBrainsToReference(brain_file_path,
                              reference_brain=meanbrain,
                              type_of_transform='SyN')

# %% tweak individual brain registrations that are problematic
# Problematic:
#   20210811-003 is way off
    # Split affine & Syn, then lobe mask. Looks OK now
#   20210820-005 is a little off, down by 12 & 17 esp.
    # Helps to split affine & syn only
#   20210820-009 is way off
    # Lobe mask helps. Brain missing part of 6 & 16 at top

mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format('20210824')
lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brain', mask_fn)).dataobj).astype('uint32')
lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain.spacing)

file_path = os.path.join(base_dir, 'anatomical_brains', 'TSeries-20210811-003_anatomical.nii')
individual_brain = ants.image_read(file_path)

fixed_red = ants.n4_bias_field_correction(ants.split_channels(meanbrain)[0])
moving_red = ants.n4_bias_field_correction(ants.split_channels(individual_brain)[0])

reg_aff = ants.registration(fixed=fixed_red,
                            moving=moving_red,
                            type_of_transform='Affine',
                            flow_sigma=6,
                            total_sigma=0,
                            random_seed=0,
                            aff_sampling=32,
                            grad_step=0.05,
                            reg_iterations=[250, 100, 50],
                            mask=lobe_mask)

registerBrainsToReference(file_path,
                          reference_brain=meanbrain,
                          type_of_transform='SyNOnly',
                          flow_sigma=6,
                          total_sigma=0,
                          initial_transform=reg_aff['fwdtransforms'][0],
                          do_bias_correction=True,
                          mask=lobe_mask)

# %%
# check the overlay
series_name = os.path.split(file_path)[-1].split('_')[0]
# Make save paths for transforms
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_anatomical', series_name)
brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
slices = [2, 5, 10, 20, 30, 40, 44]

fh, ax = plt.subplots(2, len(slices), figsize=(16, 4))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, s in enumerate(slices):
    ax[0, s_ind].imshow(ants.split_channels(meanbrain)[0][:, :, s].T, cmap='Reds')
    ax[1, s_ind].imshow(ind_red[:, :, s].T, cmap='Reds')



# %% Compare all registrations to eachother

slices = [10, 20, 30, 40]


fh, ax = plt.subplots(len(file_paths), len(slices), figsize=(12, 18))
[x.set_axis_off() for x in ax.ravel()]
for f_ind, fp in enumerate(file_paths):
    series_name = os.path.split(fp)[-1].split('_')[0]
    transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_anatomical', series_name)
    brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
    ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
    for s_ind, s in enumerate(slices):
        ax[f_ind, s_ind].imshow(ind_red[:, :, s].T, cmap='Reds')

        if s_ind == 0:
            ax[f_ind, s_ind].set_title(series_name)


# %%
