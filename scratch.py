from visanalysis.plugin import bruker
from visanalysis.util import h5io
import os
import glob


sync_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync'
datafile_dir = os.path.join(sync_dir, 'datafiles')


fps = glob.glob(os.path.join(datafile_dir, '*.hdf5'))

for experiment_filepath in fps:
    plug = bruker.BrukerPlugin()

    # # Add analysis flags to each series
    # for sn in plug.getSeriesNumbers(file_path=experiment_filepath):
    #     h5io.deleteSeriesAttribute(file_path=experiment_filepath,
    #                                series_number=sn,
    #                                attr_key='video_trim',
    #                                )
    #
    # print('Attached tags to {}'.format(experiment_filepath))

# %%
import ants
import numpy as np

sync_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync'

functional_fn = 'TSeries-20220318-008'
fp = os.path.join(sync_dir, 'overlays', '{}_masked.nii'.format(functional_fn))

merged = ants.image_read(fp)
ants.plot(ants.split_channels(merged)[0], ants.split_channels(merged)[2],
          cmap='Reds', overlay_cmap='Blues', overlay_alpha=0.5,
          axis=2, reorient=False,
          title=functional_fn, slices=np.arange(0, 12))



# %%
