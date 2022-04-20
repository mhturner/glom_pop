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
import h5py
import functools

experiment_filepath
series_number = 3

h5io.seriesExists(experiment_filepath, 8)

with h5py.File(experiment_filepath, 'r+') as experiment_file:
    find_partial = functools.partial(h5io.find_series, sn=series_number)
    epoch_run_group = experiment_file.visititems(find_partial)
    print(epoch_run_group)
    \

 #%%

 glob.glob(os.path.join(datafile_dir, 'asdgfas'))
