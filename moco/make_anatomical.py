import argparse
import numpy as np
import nibabel as nib
from visanalysis.plugin import bruker
import ants
import time

t0 = time.time()

parser = argparse.ArgumentParser(description='Make time-average anatomical scan from motion corrected anatomy series')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')
args = parser.parse_args()


metadata = bruker.getMetaData(args.file_base_path)
spacing = [float(metadata['micronsPerPixel_XAxis']),
           float(metadata['micronsPerPixel_YAxis']),
           float(metadata['micronsPerPixel_ZAxis'])]
print('Loaded metadata from {}'.format(args.file_base_path))
print('Spacing (um/pix): {}'.format(spacing))

meanbrain = np.asarray(nib.load(args.file_base_path+'_reg.nii').dataobj, dtype='uint16').mean(axis=3)
print('Made meanbrain from {}'.format(args.file_base_path+'_reg.nii'))
print('Meanbrain shape: {}'.format(meanbrain.shape))

# ANTs images:
ch1 = ants.from_numpy(meanbrain[:, :, :, 0], spacing=spacing)
ch2 = ants.from_numpy(meanbrain[:, :, :, 1], spacing=spacing)

merged = ants.merge_channels([ch1, ch2])
save_path = '{}_anatomical.nii'.format(args.file_base_path)
# Note saving as ANTs image here (32 bit)
ants.image_write(merged, save_path)

print('Saved registered meanbrain to {}. Total time = {:.1f}'.format(save_path, time.time()-t0))
