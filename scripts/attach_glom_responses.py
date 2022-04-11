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
from visanalysis.plugin import bruker
from visanalysis.analysis import imaging_data

from glom_pop import dataio, alignment

target_datafiles = ['2022-04-07.hdf5']

meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')

data_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker'
sync_dir = '/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync'
datafile_dir = os.path.join(sync_dir, 'datafiles')
transform_directory = os.path.join(sync_dir, 'transforms')

today = datetime.datetime.today().strftime('%Y%m%d')
# %%

# Load master meanbrain
meanbrain = ants.image_read(os.path.join(sync_dir, 'mean_brain', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

# load transformed mask, in meanbrain space
fp_mask = os.path.join(transform_directory, 'meanbrain_template', 'glom_mask_reg2meanbrain.nii')
glom_mask_2_meanbrain = ants.image_read(fp_mask).numpy()

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
vals = np.unique(glom_mask_2_meanbrain)[1:].astype('int')  # exclude first val (=0, not a glom)
names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']

# convert mask back to ANTs image
glom_mask_2_meanbrain = ants.from_numpy(glom_mask_2_meanbrain, spacing=meanbrain_red.spacing)

plug = bruker.BrukerPlugin()
for df in target_datafiles:
    experiment_filepath = os.path.join(datafile_dir, df)

    for sn in plug.getSeriesNumbers(file_path=experiment_filepath):

        overall_t0 = time.time()
        ID = imaging_data.ImagingDataObject(file_path=experiment_filepath,
                                            series_number=sn,
                                            quiet=True)

        if ID.getRunParameters('include_in_analysis'):
            print('Starting series {}:{}'.format(ID.file_path, ID.series_number))
            functional_fn = 'TSeries-' + os.path.split(ID.file_path)[-1].split('.')[0].replace('-', '') + '-' + str(ID.series_number).zfill(3)
            anatomical_fn = 'TSeries-' + ID.getRunParameters('anatomical_brain')

            # # # Load anatomical scan # # #
            anat_filepath = os.path.join(sync_dir, 'anatomical_brains', anatomical_fn + '_anatomical.nii')
            red_brain = ants.split_channels(ants.image_read(anat_filepath))[0]

            date_str = functional_fn.split('-')[1]
            series_number = ID.series_number

            # # # Load anatomical scan # # #
            anat_filepath = os.path.join(sync_dir, 'anatomical_brains', anatomical_fn + '_anatomical.nii')
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
            save_path = os.path.join(sync_dir, 'overlays', '{}_masked.nii'.format(functional_fn))
            ants.image_write(merged, save_path)

            # Load functional (green) brain series
            green_brain = np.asanyarray(nib.load(fxn_filepath + '_reg.nii').dataobj)[..., 1]

            # yank out glom responses
            # glom_responses: mean response across all voxels in each glom
            # shape = glom ID x Time
            glom_responses = alignment.get_glom_responses(green_brain,
                                                          glom_mask_2_fxn.numpy(),
                                                          mask_values=vals)

            # voxel_responses: list of arrays, each is all individual voxel responses for that glom
            # list of len=gloms, each with array nvoxels x time
            voxel_responses = alignment.get_glom_voxel_responses(green_brain,
                                                                 glom_mask_2_fxn.numpy(),
                                                                 mask_values=vals)

            # attach all this to the hdf5 file
            meanbrain = dataio.merge_channels(fxn_red.numpy(), fxn_green.numpy())
            dataio.attach_responses(file_path=experiment_filepath,
                                    series_number=series_number,
                                    mask=glom_mask_2_fxn.numpy(),
                                    meanbrain=meanbrain,
                                    responses=glom_responses,
                                    mask_vals=vals,
                                    response_set_name='glom',
                                    voxel_responses=voxel_responses)

            print('Done. Attached responses to {} (total: {:.1f} sec)'.format(experiment_filepath, time.time()-overall_t0))

            print('-----------------------')
