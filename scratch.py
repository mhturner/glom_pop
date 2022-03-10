from visanalysis.plugin import bruker
from visanalysis.util import registration
import ants
import time

t0 = time.time()
fp_base = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220307/TSeries-20220307-002'

metadata = bruker.getMetaData(fp_base)
spacing = [float(metadata['micronsPerPixel_XAxis']),
           float(metadata['micronsPerPixel_YAxis']),
           float(metadata['micronsPerPixel_ZAxis'])]

ch1_mb = registration.get_ants_brain(fp_base+'_reg.nii', metadata, channel=0).mean(axis=3)
ch2_mb = registration.get_ants_brain(fp_base+'_reg.nii', metadata, channel=1).mean(axis=3)

ch1 = ants.from_numpy(ch1_mb, spacing=spacing)
ch2 = ants.from_numpy(ch2_mb, spacing=spacing)

merged = ants.merge_channels([ch1, ch2])
save_path = '{}_anatomical.nii'.format(fp_base)
# Note saving as ANTs image here (32 bit)
ants.image_write(merged, save_path)

print('Saved registered meanbrain to {}. Total time = {:.1f}'.format(save_path, time.time()-t0))
