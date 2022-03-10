from visanalysis.plugin import bruker
from visanalysis.util import h5io
import os
from glom_pop import dataio

experiment_file_directory = dataio.get_config_file()['experiment_file_directory']

expt_files = [
              '2021-08-04',
              '2021-08-11',
              '2021-08-20',
              '2021-08-25',
              '2021-11-29',
              '2021-12-02',
              '2021-12-07',
              '2021-12-08',
              '2022-03-01',
              '2022-03-07',
              '2022-03-08',
]

for experiment_file_name in expt_files:
    experiment_filepath = os.path.join(experiment_file_directory, experiment_file_name) +'.hdf5'

    plug = bruker.BrukerPlugin()

    # Add 'include_in_analysis' flag to each series
    for sn in plug.getSeriesNumbers(file_path=experiment_filepath):
        h5io.updateSeriesAttribute(file_path=experiment_filepath,
                                   series_number=sn,
                                   attr_key='include_in_analysis',
                                   attr_val=True)

        h5io.updateSeriesAttribute(file_path=experiment_filepath,
                                   series_number=sn,
                                   attr_key='anatomical_brain',
                                   attr_val='')

        h5io.updateSeriesAttribute(file_path=experiment_filepath,
                                   series_number=sn,
                                   attr_key='series_notes',
                                   attr_val='')
