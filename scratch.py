import os
import h5py
import functools

data_dir = '/Users/mhturner/CurrentData'
file_path = os.path.join(data_dir, '2021-08-04.hdf5')
series_number = 1
response_set_name = 'glom'

def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj

# %%

with h5py.File(file_path, 'r+') as experiment_file:
    find_partial = functools.partial(find_series, sn=series_number)
    epoch_run_group = experiment_file.visititems(find_partial)
    parent_roi_group = epoch_run_group.require_group('aligned_response')
    # current_roi_group = parent_roi_group.require_group(response_set_name)

    if parent_roi_group.get(response_set_name):
        print('Deleting existing group {}'.format(response_set_name))
        del parent_roi_group[response_set_name]


    experiment_file.close()


# %%

with h5py.File(file_path, 'r+') as experiment_file:
    if experiment_file['Flies/Fly2']:
        print('Deleting Flies/Fly2')
        del experiment_file['Flies/Fly2']

    experiment_file.close()
