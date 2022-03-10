"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import functools
import inspect
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import yaml
import xml.etree.ElementTree as ET

from glom_pop import util
from visanalysis.util import h5io

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Config settings  # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_config_file():
    path_to_config_file = os.path.join(inspect.getfile(util).split('glom_pop')[0], 'glom_pop', 'config.yaml')
    with open(path_to_config_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


# def get_dataset(dataset_id, only_included=True):
#     path_to_dataset_yaml = get_config_file()['dataset_yaml']
#     with open(path_to_dataset_yaml, 'r') as ymlfile:
#         data_file = yaml.safe_load(ymlfile)
#         dataset = data_file.get(dataset_id)
#
#     if only_included:
#         dataset = {entry: dataset.get(entry) for entry in dataset if dataset.get(entry).get('included')}
#     else:
#         pass
#
#     return dataset


def get_included_gloms():
    return get_config_file()['included_gloms']


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Image processing # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def merge_channels(ch1, ch2):
    """
    Merge two channel brains into single array.

    ch1, ch2: np array single channel brain (dims)

    return
        merged np array, 2 channel brain (dims, c)

    """
    return np.stack([ch1, ch2], axis=-1)  # c is last dimension


def save_transforms(registration_object, transform_dir):
    """Save transforms from ANTsPy registration."""
    os.makedirs(os.path.join(transform_dir, 'forward'), exist_ok=True)
    os.makedirs(os.path.join(transform_dir, 'inverse'), exist_ok=True)

    shutil.copy(registration_object['fwdtransforms'][0], os.path.join(transform_dir, 'forward', 'warp.nii.gz'))
    shutil.copy(registration_object['fwdtransforms'][1], os.path.join(transform_dir, 'forward', 'affine.mat'))

    shutil.copy(registration_object['invtransforms'][1], os.path.join(transform_dir, 'inverse', 'warp.nii.gz'))
    shutil.copy(registration_object['invtransforms'][0], os.path.join(transform_dir, 'inverse', 'affine.mat'))


def get_transform_list(transform_dir, direction='forward'):
    """Get transform list from directory, based on direction of transform."""
    if direction == 'forward':
        transform_list = [os.path.join(transform_dir, 'forward', 'warp.nii.gz'),
                          os.path.join(transform_dir, 'forward', 'affine.mat')]
    elif direction == 'inverse':
        transform_list = [os.path.join(transform_dir, 'inverse', 'affine.mat'),
                          os.path.join(transform_dir, 'inverse', 'warp.nii.gz')]

    return transform_list

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Bruker / Prairie View metadata functions # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_bruker_metadata(file_path):
    """
    Parse Bruker / PrairieView metadata from .xml file.

    file_path: .xml filepath
    returns
        metadata: dict
    """
    root = ET.parse(file_path).getroot()

    metadata = {}
    for child in list(root.find('PVStateShard')):
        if child.get('value') is None:
            for subchild in list(child):
                new_key = child.get('key') + '_' + subchild.get('index')
                new_value = subchild.get('value')
                metadata[new_key] = new_value

        else:
            new_key = child.get('key')
            new_value = child.get('value')
            metadata[new_key] = new_value

    metadata['version'] = root.get('version')
    metadata['date'] = root.get('date')
    metadata['notes'] = root.get('notes')

    # Get axis dims
    sequences = root.findall('Sequence')
    c_dim = len(sequences[0].findall('Frame')[0].findall('File'))  # number of channels
    x_dim = metadata['pixelsPerLine']
    y_dim = metadata['linesPerFrame']

    if root.find('Sequence').get('type') == 'TSeries Timed Element':  # Plane time series
        t_dim = len(sequences[0].findall('Frame'))
        z_dim = 1
    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element':  # Volume time series
        t_dim = len(sequences)
        z_dim = len(sequences[0].findall('Frame'))
    elif root.find('Sequence').get('type') == 'ZSeries':  # Single Z stack (anatomical)
        t_dim = 1
        z_dim = len(sequences[0].findall('Frame'))
    else:
        print('!Unrecognized series type in PV metadata!')

    metadata['image_dims'] = [int(x_dim), int(y_dim), z_dim, t_dim, c_dim]

    # get frame times
    if root.find('Sequence').get('type') == 'TSeries Timed Element':  # Plane time series
        frame_times = [float(fr.get('relativeTime')) for fr in root.find('Sequence').findall('Frame')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element':  # Volume time series
        middle_frame = int(len(root.find('Sequence').findall('Frame')) / 2)
        frame_times = [float(seq.findall('Frame')[middle_frame].get('relativeTime')) for seq in root.findall('Sequence')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    return metadata

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Interacting with hdf5 data file  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def attach_responses(file_path, series_number, mask, meanbrain, responses, mask_vals,
                     response_set_name='glom', voxel_responses=None):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        parent_roi_group = epoch_run_group.require_group('aligned_response')
        current_roi_group = parent_roi_group.require_group(response_set_name)

        overwrite_dataset(current_roi_group, 'mask', mask)
        overwrite_dataset(current_roi_group, 'response', responses)
        overwrite_dataset(current_roi_group, 'meanbrain', meanbrain)

        if voxel_responses is not None:
            for ind, vr in enumerate(voxel_responses):
                overwrite_dataset(current_roi_group, 'voxel_resp_{}'.format(mask_vals[ind]), vr)

        current_roi_group.attrs['mask_vals'] = mask_vals


def overwrite_dataset(group, name, data):
    if group.get(name):
        del group[name]
    group.create_dataset(name, data=data)


def load_responses(ID, response_set_name='glom', get_voxel_responses=False):

    response_data = {}
    with h5py.File(ID.file_path, 'r') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=ID.series_number)
        roi_parent_group = experiment_file.visititems(find_partial)['aligned_response']
        roi_set_group = roi_parent_group[response_set_name]
        response_data['response'] = roi_set_group.get("response")[:]
        response_data['mask'] = roi_set_group.get("mask")[:]
        response_data['meanbrain'] = roi_set_group.get("meanbrain")[:]
        response_data['mask_vals'] = roi_set_group.attrs['mask_vals']

        if get_voxel_responses:
            mask_vals = np.unique(response_data['mask'])[1:].astype(int)  # exclude first (0)
            voxel_responses = {}
            voxel_epoch_responses = {}
            for mv in mask_vals:
                voxel_responses[mv] = roi_set_group.get('voxel_resp_{}'.format(mv))[:].astype('float32')
                _, response_matrix = ID.getEpochResponseMatrix(voxel_responses[mv], dff=False)
                voxel_epoch_responses[mv] = response_matrix

    if get_voxel_responses:
        response_data['voxel_responses'] = voxel_responses
        response_data['voxel_epoch_responses'] = voxel_epoch_responses

    # epoch_response matrix for glom responses
    time_vector, response_matrix = ID.getEpochResponseMatrix(response_data.get('response'), dff=True)

    response_data['epoch_response'] = response_matrix  # shape = (gloms, trials, time)
    response_data['time_vector'] = time_vector

    return response_data


def get_glom_mask_decoder(mask):
    sync_dir = get_config_file()['sync_dir']
    # Load mask key for VPN types
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))

    vals = np.unique(mask)[1:]  # exclude first val (=0, not a glom)

    names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']
    return vals, names


def get_glom_name_from_val(val):
    sync_dir = get_config_file()['sync_dir']
    # Load mask key for VPN types
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
    name = vpn_types.iloc[np.where(vpn_types['Unnamed: 0'] == val)[0], 1].values[0]

    return name


def get_glom_vals_from_names(glom_names):
    sync_dir = get_config_file()['sync_dir']
    vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
    vals = np.array([vpn_types.iloc[np.where(vpn_types.vpn_types == ig)[0][0], 0] for ig in glom_names])

    return vals
