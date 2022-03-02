"""
Motion correction of functional brain
mhturner@stanford.edu
"""

import time
import nibabel as nib
import numpy as np
from visanalysis.util import registration
from visanalysis.plugin import bruker
import ants
import argparse

t0 = time.time()

# first arg: path to image series base, without .suffix
#   e.g. /path/to/imaging/data/TSeries-20210611-001
parser = argparse.ArgumentParser(description='ANTsPy based motion correction for volumetric time series')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')

parser.add_argument('--noisy-mode', dest='noisy',
                    action='store_true',
                    help='Flag to enable Moco optimized for noisy/sparse images. Slower than standard moco.')

parser.add_argument('--meanbrain-only', dest='meanbrain_only',
                    action='store_true',
                    help='Flag to save only the time-averaged brain, not the full corrected time series')

args = parser.parse_args()

print('Registering brain file from {}'.format(args.file_base_path))

# Load metadata from bruker .xml file
metadata = bruker.getMetaData(args.file_base_path)
print('Loaded metadata from {}'.format(args.file_base_path))

# Load brain images
ch1 = registration.get_ants_brain(args.file_base_path + '_channel_1.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(args.file_base_path + '_channel_1.nii', ch1.shape))
ch2 = registration.get_ants_brain(args.file_base_path + '_channel_2.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(args.file_base_path + '_channel_2.nii', ch2.shape))


if args.noisy:
    print('Noisy motion correction...')
    ch1_corrected, ch2_corrected = registration.motion_correct_noisy(reference_channel=ch1,
                                                                     yoked_channel=ch2,
                                                                     reference_frames=100,
                                                                     type_of_transform='Rigid',
                                                                     flow_sigma=3,
                                                                     total_sigma=0,
                                                                     filter_transforms=False,
                                                                     smoothing_sigma=[1.0, 1.0, 0.0, 2.0]
                                                                     )

else:
    print('Standard motion correction...')
    ch1_corrected, ch2_corrected = registration.motion_correct(reference_channel=ch1,
                                                               yoked_channel=ch2,
                                                               reference_frames=100,
                                                               type_of_transform='Rigid',
                                                               flow_sigma=3,
                                                               total_sigma=0
                                                               )

merged_corrected = registration.merge_channels(ch1_corrected, ch2_corrected)

if args.meanbrain_only:
    spacing = [float(metadata['micronsPerPixel_XAxis']),
               float(metadata['micronsPerPixel_YAxis']),
               float(metadata['micronsPerPixel_ZAxis'])]

    del ch1_corrected, ch2_corrected  # save memory

    # Time average, shape = xyzc
    meanbrain = merged_corrected.mean(axis=3)

    # ANTs images:
    ch1 = ants.from_numpy(meanbrain[:, :, :, 0], spacing=spacing)
    ch2 = ants.from_numpy(meanbrain[:, :, :, 1], spacing=spacing)

    merged = ants.merge_channels([ch1, ch2])
    save_path = '{}_anatomical.nii'.format(args.file_base_path)
    # Note saving as ANTs image here (32 bit)
    ants.image_write(merged, save_path)

    print('Saved registered meanbrain to {}. Total time = {:.1f}'.format(save_path, time.time()-t0))

else:
    # Save registered, merged .nii
    nifti1_limit = (2**16 / 2)
    if np.any(np.array(merged_corrected.shape) >= nifti1_limit):  # Need to save as nifti2
        nib.save(nib.Nifti2Image(merged_corrected, np.eye(4)), args.file_base_path + '_reg.nii')
    else:  # Nifti1 is OK
        nib.save(nib.Nifti1Image(merged_corrected, np.eye(4)), args.file_base_path + '_reg.nii')
    print('Saved registered brain to {}'.format(args.file_base_path + '_reg.nii'))
print('Total time = {:.1f}'.format(time.time()-t0))
