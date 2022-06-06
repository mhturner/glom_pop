

# %% Eg video of Gcamp resp and behavior...
import nibabel as nib
import os
from skimage.filters import gaussian
import numpy as np
import ants


eg_series = ('2022-04-12', 1)

data_dir = '/Users/mhturner/CurrentData/20220412'
ch2 = np.asarray(nib.load(os.path.join(data_dir, 'TSeries-20220412-001_reg.nii')).dataobj, dtype='uint16')[:, :, :, :, 1]
ch2.shape
ch2_smooth = gaussian(ch2, sigma=(1, 1, 0, 3))
ants.image_write(ants.from_numpy(ch2_smooth), os.path.join(data_dir, 'TSeries-20220412-001_smooth_ch2.nii'))
# %%
