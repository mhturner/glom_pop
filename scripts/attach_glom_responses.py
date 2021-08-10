"""
Attach glom responses to datafile based on aligned glom map

(1) Align ANAT scan to MEANBRAIN (AM)
(2) Align FXN scan to fly's own ANAT scan (FA)
(3) Bridge meanbrain-aligned glom map into FXN: Inverse AM + Inverse FA
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


#                   (Time series, associated anatomical z series for this fly)
brain_file_sets = [('TSeries-20210804-001', 'ZSeries-20210804-001'),
                   ('TSeries-20210804-002', 'ZSeries-20210804-001'),
                   ('TSeries-20210804-004', 'ZSeries-20210804-004'),
                   ('TSeries-20210804-005', 'ZSeries-20210804-004'),
                   ('TSeries-20210804-007', 'ZSeries-20210804-007'),
                   ('TSeries-20210804-008', 'ZSeries-20210804-007'),
                   ]

meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210805')

data_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker'
base_dir = '/oak/stanford/groups/trc/data/Max/Analysis/glom_pop'
datafile_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/DataFiles'

today = datetime.datetime.today().strftime('%Y%m%d')
# %%

# Load master meanbrain
reference_fn = 'ZSeries-20210804-001'
metadata = dataio.get_bruker_metadata(os.path.join(base_dir, 'anatomical_brains', reference_fn) + '.xml')
meanbrain_red = dataio.get_ants_brain(os.path.join(base_dir, 'mean_brains', meanbrain_fn), metadata, channel=0)

# Load glom map, in meanbrain space
mask_fp = os.path.join(base_dir, 'aligned', 'glom_mask_reg2meanbrain.nii')
glom_mask_2_meanbrain = np.asanyarray(nib.load(mask_fp).dataobj).astype('uint32')
# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))
# Filter glom map s.t. only big gloms are included
glom_size_threshold = 300
glom_mask_2_meanbrain = alignment.filterGlomMask(glom_mask_2_meanbrain, glom_size_threshold)
vals = np.unique(glom_mask_2_meanbrain)[1:]  # exclude first val (=0, not a glom)
names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']

# convert to ants image
glom_mask_2_meanbrain = ants.from_numpy(glom_mask_2_meanbrain, spacing=meanbrain_red.spacing)

for bf in brain_file_sets:
    t0 = time.time()
    functional_fn = bf[0]
    anatomical_fn = bf[1]

    print('Starting brain from {}'.format(functional_fn))

    date_str = functional_fn.split('-')[1]
    experiment_file_name = '{}-{}-{}'.format(date_str[0:4], date_str[4:6], date_str[6:8])
    h5_filepath = os.path.join(datafile_dir, experiment_file_name + '.hdf5')
    series_number = int(functional_fn.split('-')[-1])

    # # # Load anatomical scan # # #
    anat_filepath = os.path.join(data_dir, date_str, anatomical_fn)
    metadata = dataio.get_bruker_metadata(anat_filepath + '.xml')
    red_brain = dataio.get_ants_brain(anat_filepath + '_channel_1.nii', metadata)  # xyz

    # # # Compute transform from ANAT -> MEANBRAIN # # #
    reg_AM = ants.registration(meanbrain_red,
                               red_brain,
                               type_of_transform='SyN',
                               flow_sigma=3,
                               total_sigma=0)

    # # # Apply inverse transform to glom mask # # #
    glom_mask_2_anat = ants.apply_transforms(fixed=red_brain,
                                             moving=glom_mask_2_meanbrain,
                                             transformlist=reg_AM['invtransforms'],
                                             interpolator='nearestNeighbor',
                                             defaultvalue=0)

    print('Applied inverse transform MEANBRAIN -> ANAT to glom_mask: {} ({} sec)'.format(anatomical_fn, time.time()-t0))

    # # # Compute transform from FXN -> ANAT # # #
    fxn_filepath = os.path.join(data_dir, date_str, functional_fn)
    metadata_fxn = dataio.get_bruker_metadata(fxn_filepath + '.xml')
    fxn_red = dataio.get_time_averaged_brain(dataio.get_ants_brain(fxn_filepath + '_reg.nii', metadata_fxn, channel=0))  # xyz
    fxn_green = dataio.get_time_averaged_brain(dataio.get_ants_brain(fxn_filepath + '_reg.nii', metadata_fxn, channel=1))  # xyz
    reg_FA = ants.registration(red_brain,
                               fxn_red,
                               type_of_transform='SyN',
                               flow_sigma=3,
                               total_sigma=0)

    # # # Apply inverse transform to glom mask # # #
    glom_mask_2_fxn = ants.apply_transforms(fixed=fxn_red,
                                            moving=glom_mask_2_anat,
                                            transformlist=reg_FA['invtransforms'],
                                            interpolator='nearestNeighbor',
                                            defaultvalue=0)

    print('Applied inverse transform ANAT -> FXN to glom_mask: {} ({} sec)'.format(functional_fn, time.time()-t0))

    # Load functional (green) brain series
    green_brain = dataio.get_ants_brain(fxn_filepath + '_reg.nii', metadata_fxn, channel=1)  # xyzt

    # yank out glom responses
    glom_responses = alignment.getGlomResponses(green_brain,
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
                           response_set_name='glom_{}'.format(today))

    print('Attached responses to {} ({} sec)'.format(h5_filepath, time.time()-t0))

    print('-----------------------')
