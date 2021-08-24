"""
Make meanbrain from motion corrected time series

mhturner@stanford.edu
"""
import sys
import time
from visanalysis.util import registration
import os
import ants

t0 = time.time()
# first arg: path to image series base, without .suffix
#   e.g. /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20210611/TSeries-20210611-001
file_base_path = sys.argv[1]
print('Making moco meanbrain from {}'.format(file_base_path))
fn = os.path.split(file_base_path)[-1]
date = os.path.split(file_base_path)[-1].split('-')[1]

# Load metadata from bruker .xml file
metadata = registration.get_bruker_metadata(file_base_path + '.xml')
spacing = [float(metadata['micronsPerPixel_XAxis']),
           float(metadata['micronsPerPixel_YAxis']),
           float(metadata['micronsPerPixel_ZAxis'])]
print('Loaded metadata from {}'.format(file_base_path + '.xml'))

# Load brain images
ch1 = registration.get_ants_brain(file_base_path + '_channel_1.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(file_base_path + '_channel_1.nii', ch1.shape))
ch2 = registration.get_ants_brain(file_base_path + '_channel_2.nii', metadata, channel=0)
print('Loaded {}, shape={}'.format(file_base_path + '_channel_2.nii', ch2.shape))

# Register both channels to channel 1
# shape = xyztc
merged = registration.register_two_channels_to_red(ch1, ch2, spatial_dims=len(ch1.shape) - 1, reference_frames=10)

# shape = xyzc
meanbrain = merged.mean(axis=3)

ch1 = ants.from_numpy(meanbrain[:, :, :, 0], spacing=spacing)
ch2 = ants.from_numpy(meanbrain[:, :, :, 1], spacing=spacing)

merged = ants.merge_channels([ch1, ch2])
save_path = os.path.join(os.path.split(file_base_path)[0], '{}_anatomical.nii'.format(fn))
ants.image_write(merged, save_path)

print('Saved registered brain to {}. Total time = {:.1f}'.format(save_path, time.time()-t0))
