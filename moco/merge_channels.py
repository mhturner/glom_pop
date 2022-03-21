import os
import argparse
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Merge 2 nii series to a single nii file')
parser.add_argument('series_base', type=str,
                    help='Path to ch1 + ch2 images, without suffixes, e.g. /path/to/data/moco/TSeries-20220301-001')
args = parser.parse_args()

ch1_path = args.series_base + '_channel_1_moco.nii'
ch2_path = args.series_base + '_channel_2_moco.nii'

merged = np.stack([np.asarray(nib.load(ch1_path).dataobj, dtype='uint16'),
                   np.asarray(nib.load(ch2_path).dataobj, dtype='uint16')], axis=-1)  # c is last dimension

# Save registered, merged .nii
save_dir = os.path.dirname(os.path.split(args.series_base)[0])  # Save it up one dir
save_path = os.path.join(save_dir, os.path.split(args.series_base)[-1] + '_reg.nii')
nifti1_limit = (2**16 / 2)
if np.any(np.array(merged.shape) >= nifti1_limit):  # Need to save as nifti2
    nib.save(nib.Nifti2Image(merged, np.eye(4)), save_path)
else:  # Nifti1 is OK
    nib.save(nib.Nifti1Image(merged, np.eye(4)), save_path)
print('Saved registered brain to {}'.format(save_path))

# Delete individual channels
os.remove(ch1_path)
os.remove(ch2_path)
