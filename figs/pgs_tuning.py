from visanalysis.analysis import imaging_data
import matplotlib.pyplot as plt
import numpy as np
import os

from glom_pop import dataio


experiment_file_directory = '/Users/mhturner/CurrentData'
experiment_file_name = '2021-08-04'
series_number = 1

file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

# ImagingDataObject wants a path to an hdf5 file and a series number from that file
ID = imaging_data.ImagingDataObject(file_path,
                                    series_number,
                                    quiet=False)

# %%
ID.getStimulusTiming(plot_trace_flag=True)

# %%

response_data = dataio.loadResponses(ID, response_set_name='glom_20210809')
