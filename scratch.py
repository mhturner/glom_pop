from visanalysis.util import h5io
import os
import h5py

data_dir = '/Users/mhturner/CurrentData/'
file_path = os.path.join(data_dir, '2022-03-28.hdf5')

with h5py.File(file_path, 'r+') as h5file:

    source = '/Flies/Fly1/'
    dest = '/Flies/Fly2'
    grp = h5file[source]
    print(grp)
    h5file.copy(source, dest, name='Fly2')
