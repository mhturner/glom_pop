import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from visanalysis.analysis import shared_analysis

from glom_pop import dataio, model, util

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']


# First include all gloms and all flies
leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

target_series_metadata = {'protocol_ID': 'ExpandingMovingSpot',
                          'include_in_analysis': True,
                          'diameter': 15.0,
                          }
matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'LC18',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  # target_series_metadata=target_series_metadata,
                                                  target_groups=['aligned_response', 'behavior'])


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
ch2_smooth = gaussian(ch2, sigma=(2, 2, 0, 1))
ants.image_write(ants.from_numpy(ch2_smooth), os.path.join(data_dir, 'TSeries-20220412-001_smooth_ch2.nii'))
# %%
