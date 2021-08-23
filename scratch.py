# # %%
#
# import os
# import ants
# import nibabel as nib
# import numpy as np
# from visanalysis.util import registration
#
# import glob
# # %%
#
# base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
#
# fns = glob.glob(os.path.join(base_dir, 'anatomical_brains', '*.xml'))
# fns
#
# fns[0].split('.')[0]
#
# for fn in fns:
#     fn_base = fn.split('.')[0]
#
#     # Load metadata from bruker .xml file
#     metadata = registration.get_bruker_metadata(fn_base + '.xml')
#     spacing = [float(metadata['micronsPerPixel_XAxis']),
#                float(metadata['micronsPerPixel_YAxis']),
#                float(metadata['micronsPerPixel_ZAxis'])]
#
#     # Load brain images
#     nib_brain = np.asanyarray(nib.load(fn_base + '_anatomical.nii').dataobj).astype('uint32')
#     ch1 = ants.from_numpy(nib_brain[:, :, :, 0], spacing=spacing)
#     ch2 = ants.from_numpy(nib_brain[:, :, :, 1], spacing=spacing)
#
#     merged = ants.merge_channels([ch1, ch2])
#     ants.image_write(merged, fn_base + '_anatomical.nii')
