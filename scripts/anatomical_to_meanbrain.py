"""
Register anatomical scans to pre-computed meanbrain

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import nibabel as nib
import ants
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from glom_pop import dataio

sync_dir = dataio.get_config_file()['sync_dir']

# %% Fxn def'n


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


# %% Register each brain to final meanbrain and save these transforms

# Load meanbrain
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')
meanbrain = ants.image_read(os.path.join(sync_dir, 'mean_brain', meanbrain_fn))

# Register all anatomical brains to meanbrain
# file_paths = glob.glob(os.path.join(sync_dir, 'anatomical_brains', '*_anatomical.nii'))

# Register select brains
file_names = [
              'TSeries-20220404-005_anatomical.nii',
              'TSeries-20220404-009_anatomical.nii',
              ]
file_paths = [os.path.join(sync_dir, 'anatomical_brains', x) for x in file_names]
print(file_paths)
# %% Run through all brains in dir
# Register each anatomical brain to the meanbrain
# For each anatomical brain, saves:
#   -forward (anat -> meanbrain) & inverse (meanbrain -> anat) transforms

for brain_file_path in file_paths:
    registerBrainsToReference(brain_file_path,
                              reference_brain=meanbrain,
                              type_of_transform='SyN')

# %% tweak individual brain registrations that are problematic
# Helps to split affine & syn only

mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format('20210824')
lobe_mask = np.asanyarray(nib.load(os.path.join(sync_dir, 'mean_brain', mask_fn)).dataobj).astype('uint32')
lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain.spacing)

file_path = os.path.join(sync_dir, 'anatomical_brains', 'TSeries-20210820-009_anatomical.nii')
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

# %% for above...
# check the overlay

series_name = os.path.split(file_path)[-1].split('_')[0]
# Make save paths for transforms
transform_dir = os.path.join(sync_dir, 'transforms', 'meanbrain_anatomical', series_name)
brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
slices = [2, 5, 10, 20, 30, 40, 44]

fh, ax = plt.subplots(2, len(slices), figsize=(16, 4))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, s in enumerate(slices):
    ax[0, s_ind].imshow(ants.split_channels(meanbrain)[0][:, :, s].T, cmap='Reds')
    ax[1, s_ind].imshow(ind_red[:, :, s].T, cmap='Reds')

# %% Compare all registrations to meanbrain

for f_ind, fp in enumerate(file_paths):
    series_name = os.path.split(fp)[-1].split('_')[0]
    transform_dir = os.path.join(sync_dir, 'transforms', 'meanbrain_anatomical', series_name)
    brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
    ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
    ants.plot(ants.split_channels(meanbrain)[0], ind_red,
              cmap='Reds', overlay_cmap='Greens', axis=2, reorient=False,
              title=series_name)


# %% Compare all registrations to eachother

slices = [10, 20, 30, 40]


fh, ax = plt.subplots(len(file_paths), len(slices), figsize=(12, 18))
[x.set_axis_off() for x in ax.ravel()]
for f_ind, fp in enumerate(file_paths):
    series_name = os.path.split(fp)[-1].split('_')[0]
    transform_dir = os.path.join(sync_dir, 'transforms', 'meanbrain_anatomical', series_name)
    brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
    ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
    for s_ind, s in enumerate(slices):
        ax[f_ind, s_ind].imshow(ind_red[:, :, s].T, cmap='Reds')

        if s_ind == 0:
            ax[f_ind, s_ind].set_title(series_name)


# %% Check alignment with meanbrain at some zoomed-in gloms
cmap = 'binary_r'
dx = 100
dy = 60

meanbrain_red = ants.split_channels(meanbrain)[0]
meanbrain_red
brain_directory = os.path.join(sync_dir, 'anatomical_brains')
# file_paths = glob.glob(os.path.join(brain_directory, '*_anatomical.nii'))

boxes = [(55, 10), (30, 110), (120, 5)]
zs = [10, 39, 39]

for b_ind, box in enumerate(boxes):
    z = zs[b_ind]
    fh, ax = plt.subplots(3, 6, figsize=(15, 6))
    ax = ax.ravel()
    [x.set_axis_off() for x in ax.ravel()]
    ax[0].imshow(meanbrain_red[box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap=cmap)
    ax[0].set_title('Mean', fontsize=11, fontweight='bold')

    [x.set_xticks([]) for x in ax]
    [x.set_yticks([]) for x in ax]
    [x.axhline(dy/2, color='w') for x in ax]
    [x.axvline(dx/2, color='w') for x in ax]
    [s.set_linewidth(2) for s in ax[0].spines.values()]

    for f_ind, fp in enumerate(file_paths):
        series_name = os.path.split(fp)[-1].split('_')[0]
        transform_dir = os.path.join(sync_dir, 'transforms', 'meanbrain_anatomical', series_name)
        brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
        ind_red = ants.split_channels(ants.image_read(brain_fp))[0]

        ax[f_ind+1].imshow(ind_red[box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap=cmap)

        ax[f_ind+1].set_title(series_name, fontsize=11)
