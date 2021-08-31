"""
Attach glom responses to datafile based on aligned glom map

(1) Load pre-computed alignment from ANAT scan to MEANBRAIN (AM)
(2) Align FXN scan to fly's own ANAT scan (FA)
(3) Bridge meanbrain-aligned glom map into FXN
(4) Use bridged glom map to pull out glom responses, and attach to visprotocol .hdf5 file


FXN -FA-> ANAT -AM-> MEANBRAIN <- JRC2018 <- GLOM MAP

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import os
import ants
import numpy as np
import time
import nibabel as nib
import pandas as pd
import datetime

from glom_pop import dataio, alignment

#                   (Time series, associated anatomical series for this fly)
brain_file_sets = [
                   ('TSeries-20210804-001', 'TSeries-20210804-003'),
                   ('TSeries-20210804-002', 'TSeries-20210804-003'),
                   ('TSeries-20210804-004', 'TSeries-20210804-006'),
                   ('TSeries-20210804-005', 'TSeries-20210804-006'),
                   ('TSeries-20210804-007', 'TSeries-20210804-009'),
                   ('TSeries-20210804-008', 'TSeries-20210804-009'),

                   ('TSeries-20210811-001', 'TSeries-20210811-003'),
                   ('TSeries-20210811-002', 'TSeries-20210811-003'),
                   ('TSeries-20210811-004', 'TSeries-20210811-006'),
                   ('TSeries-20210811-005', 'TSeries-20210811-006'),
                   ('TSeries-20210811-007', 'TSeries-20210811-009'),
                   ('TSeries-20210811-008', 'TSeries-20210811-009'),

                   ('TSeries-20210820-002', 'TSeries-20210820-005'),
                   ('TSeries-20210820-003', 'TSeries-20210820-005'),
                   ('TSeries-20210820-004', 'TSeries-20210820-005'),
                   ('TSeries-20210820-006', 'TSeries-20210820-009'),
                   ('TSeries-20210820-008', 'TSeries-20210820-009'),

                   ('TSeries-20210825-001', 'TSeries-20210825-004'),
                   ('TSeries-20210825-002', 'TSeries-20210825-004'),
                   ('TSeries-20210825-003', 'TSeries-20210825-004'),
                   ('TSeries-20210825-009', 'TSeries-20210825-012'),
                   ('TSeries-20210825-010', 'TSeries-20210825-012'),
                   ('TSeries-20210825-011', 'TSeries-20210825-012'),

                   ]

meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')

data_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker'
base_dir = '/oak/stanford/groups/trc/data/Max/Analysis/glom_pop'
datafile_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/DataFiles'

transform_directory = os.path.join(base_dir, 'transforms')

today = datetime.datetime.today().strftime('%Y%m%d')
# %%

# Load master meanbrain
meanbrain = ants.image_read(os.path.join(base_dir, 'mean_brain', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

# load transformed mask, in meanbrain space
fp_mask = os.path.join(transform_directory, 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')
glom_mask_2_meanbrain = ants.image_read(fp_mask).numpy()

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))
# Filter glom map s.t. only big gloms are included
glom_size_threshold = 350
glom_mask_2_meanbrain = alignment.filterGlomMask(glom_mask_2_meanbrain, glom_size_threshold)
vals = np.unique(glom_mask_2_meanbrain)[1:].astype('int')  # exclude first val (=0, not a glom)
names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']

# convert mask back to ANTs image
glom_mask_2_meanbrain = ants.from_numpy(glom_mask_2_meanbrain, spacing=meanbrain_red.spacing)

for bf in brain_file_sets:
    functional_fn = bf[0]
    anatomical_fn = bf[1]
    overall_t0 = time.time()
    print('Starting brain from {}'.format(functional_fn))

    date_str = functional_fn.split('-')[1]
    experiment_file_name = '{}-{}-{}'.format(date_str[0:4], date_str[4:6], date_str[6:8])
    h5_filepath = os.path.join(datafile_dir, experiment_file_name + '.hdf5')
    series_number = int(functional_fn.split('-')[-1])

    # # # Load anatomical scan # # #
    anat_filepath = os.path.join(base_dir, 'anatomical_brains', anatomical_fn + '_anatomical.nii')
    red_brain = ants.split_channels(ants.image_read(anat_filepath))[0]

    # # # (1) Transform map from MEANBRAIN -> ANAT # # #
    t0 = time.time()
    # Pre-computed is anat->meanbrain, so we want the inverse transform
    transform_dir = os.path.join(transform_directory, 'meanbrain_anatomical', anatomical_fn)
    transform_list = dataio.get_transform_list(transform_dir, direction='inverse')

    # Apply inverse transform to glom mask
    glom_mask_2_anat = ants.apply_transforms(fixed=red_brain,
                                             moving=glom_mask_2_meanbrain,
                                             transformlist=transform_list,
                                             interpolator='genericLabel',
                                             defaultvalue=0)

    print('Applied inverse transform from ANAT -> MEANBRAIN to glom mask ({:.1f} sec)'.format(time.time()-t0))

    # # # (2) Transform from ANAT -> FXN (within fly) # # #
    t0 = time.time()
    fxn_filepath = os.path.join(data_dir, date_str, functional_fn)
    metadata_fxn = dataio.get_bruker_metadata(fxn_filepath + '.xml')

    spacing = [float(metadata_fxn.get('micronsPerPixel_XAxis', 0)),
               float(metadata_fxn.get('micronsPerPixel_YAxis', 0)),
               float(metadata_fxn.get('micronsPerPixel_ZAxis', 0))]
    # load brain, average over all frames
    nib_brain = np.asanyarray(nib.load(fxn_filepath + '_reg.nii').dataobj).mean(axis=3)
    fxn_red = ants.from_numpy(nib_brain[:, :, :, 0], spacing=spacing)  # xyz
    fxn_green = ants.from_numpy(nib_brain[:, :, :, 1], spacing=spacing)  # xyz

    print('Loaded fxnal meanbrains ({:.1f} sec)'.format(time.time()-t0))
    t0 = time.time()
    reg_FA = ants.registration(fxn_red,
                               red_brain,
                               type_of_transform='Rigid',  # Within-animal, rigid reg is OK
                               flow_sigma=3,
                               total_sigma=0)

    # # # Apply inverse transform to glom mask # # #
    glom_mask_2_fxn = ants.apply_transforms(fixed=fxn_red,
                                            moving=glom_mask_2_anat,
                                            transformlist=reg_FA['fwdtransforms'],
                                            interpolator='genericLabel',
                                            defaultvalue=0)

    print('Computed transform from ANAT -> FXN & applied to glom mask ({:.1f} sec)'.format(time.time()-t0))

    # Save multichannel overlay image in fxn space: red, green, mask
    merged = ants.merge_channels([fxn_red, fxn_green, glom_mask_2_fxn])
    save_path = os.path.join(base_dir, 'overlays', '{}_masked.nii'.format(functional_fn))
    ants.image_write(merged, save_path)

    # Load functional (green) brain series
    green_brain = np.asanyarray(nib.load(fxn_filepath + '_reg.nii').dataobj)[..., 1]

    # yank out glom responses
    glom_responses = alignment.getGlomResponses(green_brain,
                                                glom_mask_2_fxn.numpy(),
                                                mask_values=vals)

    voxel_responses = alignment.getGlomVoxelResponses(green_brain,
                                                      glom_mask_2_fxn.numpy(),
                                                      mask_values=vals)

    # attach all this to the hdf5 file
    meanbrain = dataio.merge_channels(fxn_red.numpy(), fxn_green.numpy())
    dataio.attachResponses(file_path=h5_filepath,
                           series_number=series_number,
                           mask=glom_mask_2_fxn.numpy(),
                           meanbrain=meanbrain,
                           responses=glom_responses,
                           mask_vals=vals,
                           response_set_name='glom',
                           voxel_responses=voxel_responses)

    print('Done. Attached responses to {} (total: {:.1f} sec)'.format(h5_filepath, time.time()-overall_t0))

    print('-----------------------')
