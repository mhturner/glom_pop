import h5py
import argparse
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Convert and merge 2 h5 channels to a single nii file')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')

args = parser.parse_args()


def convert_h5_to_array(h5_path):
    with h5py.File(h5_path, 'r+') as h5_file:
        array = h5_file.get("data")[:].astype('uint16')
    return array


ch1_path = '{}_channel_1_moco.h5'.format(args.file_base_path)
ch1_array = convert_h5_to_array(ch1_path)

ch2_path = '{}_channel_2_moco.h5'.format(args.file_base_path)
ch2_array = convert_h5_to_array(ch2_path)

merged = np.stack([ch1_array, ch2_array], axis=-1)  # c is last dimension

# Save registered, merged .nii
save_path = '{}_reg.nii'.format(args.file_base_path)
nifti1_limit = (2**16 / 2)
if np.any(np.array(merged.shape) >= nifti1_limit):  # Need to save as nifti2
    nib.save(nib.Nifti2Image(merged, np.eye(4)), save_path)
else:  # Nifti1 is OK
    nib.save(nib.Nifti1Image(merged, np.eye(4)), save_path)
print('Saved registered brain to {}'.format(save_path))
