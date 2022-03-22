import numpy as np
import os



video_dir = '/Users/mhturner/CurrentData/

date_dir = '20220318'

series_dir = 'Series006'



# %%

import numpy as np
import os
from visanalysis.plugin import bruker
from visanalysis.util import h5io
from visanalysis.analysis import imaging_data

plug = bruker.BrukerPlugin()

# data_dir = '/oak/stanford/groups/trc/data/Max/ImagingData/Bruker'
sync_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/sync'
datafile_dir = os.path.join(sync_dir, 'datafiles')

target_datafiles = ['2022-03-17.hdf5',
                    '2022-03-18.hdf5']
for df in target_datafiles:
    experiment_filepath = os.path.join(datafile_dir, df)

    for sn in plug.getSeriesNumbers(file_path=experiment_filepath):
        ID = imaging_data.ImagingDataObject(file_path=experiment_filepath,
                                            series_number=sn,
                                            quiet=True)

        if ID.getRunParameters('include_in_analysis'):
            functional_fn = 'TSeries-' + os.path.split(ID.file_path)[-1].split('.')[0].replace('-','') + '-' + str(ID.series_number).zfill(3)
            anatomical_fn = 'TSeries-' + ID.getRunParameters('anatomical_brain')
            print(functional_fn)
            print(anatomical_fn)
