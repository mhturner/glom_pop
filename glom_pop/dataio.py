"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import xml.etree.ElementTree as ET
import ants
import numpy as np
import functools
import h5py
import os
import pandas as pd
import shutil

from visanalysis.analysis import imaging_data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Image processing # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# def get_smooth_brain(brain, smoothing_sigma=[1.0, 1.0, 0.0, 2.0]):
#     """
#     Gaussian smooth brain.
#
#     brain: ants brain. shape = (spatial..., t)
#     smoothing_sigma: Gaussian smoothing kernel. len = rank of brain
#         Spatial dims come first. T last. Default dim is [x, y, z, t]
#
#     returns smoothed brain, ants. Same dims as input brain
#     """
#     smoothed = gaussian_filter(brain.numpy(), sigma=smoothing_sigma)
#
#     return ants.from_numpy(smoothed, spacing=brain.spacing)  # xyz
#

def get_time_averaged_brain(brain, frames=None):
    """
    Time average brain.

    brain: (spatial, t) ants brain
    frames: average from 0:n frames. Note None -> average over all time

    returns time-averaged brain, ants. Dim =  (spatial)
    """
    spacing = list(np.array(brain.spacing)[..., :-1])
    return ants.from_numpy(brain[..., 0:frames].mean(axis=len(brain.shape)-1), spacing=spacing)


# def merge_channels(ch1, ch2):
#     """
#     Merge two channel brains into single array.
#
#     ch1, ch2: np array single channel brain (dims)
#
#     return
#         merged np array, 2 channel brain (dims, c)
#
#     """
#     return np.stack([ch1, ch2], axis=-1)  # c is last dimension
#

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
#
# def get_ants_brain(filepath, metadata, channel=0, spacing=None):
#     """Load .nii brain file as ANTs image."""
#     image_dims = metadata.get('image_dims')  # xyztc
#     nib_brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint32')
#     if spacing is None:  # try to infer from the metadata file
#         spacing = [float(metadata.get('micronsPerPixel_XAxis', 0)),
#                    float(metadata.get('micronsPerPixel_YAxis', 0)),
#                    float(metadata.get('micronsPerPixel_ZAxis', 0)),
#                    float(metadata.get('sample_period', 0))]
#         spacing = [spacing[x] for x in range(4) if image_dims[x] > 1]
#
#     if image_dims[4] > 1:  # multiple channels
#         # trim to single channel
#         return ants.from_numpy(np.squeeze(nib_brain[..., channel]), spacing=spacing)
#     else:
#         return ants.from_numpy(np.squeeze(nib_brain), spacing=spacing)

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
    c_dim = len(sequences[0].findall('Frame')[0].findall('File')) # number of channels
    x_dim = metadata['pixelsPerLine']
    y_dim = metadata['linesPerFrame']

    if root.find('Sequence').get('type') == 'TSeries Timed Element': # Plane time series
        t_dim = len(sequences[0].findall('Frame'))
        z_dim = 1
    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element': # Volume time series
        t_dim = len(sequences)
        z_dim = len(sequences[0].findall('Frame'))
    elif root.find('Sequence').get('type') == 'ZSeries': # Single Z stack (anatomical)
        t_dim = 1
        z_dim = len(sequences[0].findall('Frame'))
    else:
        print('!Unrecognized series type in PV metadata!')

    metadata['image_dims'] = [int(x_dim), int(y_dim), z_dim, t_dim, c_dim]

    # get frame times
    if root.find('Sequence').get('type') == 'TSeries Timed Element': # Plane time series
        frame_times = [float(fr.get('relativeTime')) for fr in root.find('Sequence').findall('Frame')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    elif root.find('Sequence').get('type') == 'TSeries ZSeries Element': # Volume time series
        middle_frame = int(len(root.find('Sequence').findall('Frame')) / 2)
        frame_times = [float(seq.findall('Frame')[middle_frame].get('relativeTime')) for seq in root.findall('Sequence')]
        metadata['frame_times'] = frame_times
        metadata['sample_period'] = np.mean(np.diff(frame_times))

    return metadata


def attachResponses(file_path, series_number, mask, meanbrain, responses, mask_vals, response_set_name='glom', voxel_responses=None):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        parent_roi_group = epoch_run_group.require_group('aligned_response')
        current_roi_group = parent_roi_group.require_group(response_set_name)

        overwriteDataSet(current_roi_group, 'mask', mask)
        overwriteDataSet(current_roi_group, 'response', responses)
        overwriteDataSet(current_roi_group, 'meanbrain', meanbrain)

        if voxel_responses is not None:
            for ind, vr in enumerate(voxel_responses):
                overwriteDataSet(current_roi_group, 'voxel_resp_{}'.format(mask_vals[ind]), vr)

        current_roi_group.attrs['mask_vals'] = mask_vals


def overwriteDataSet(group, name, data):
    if group.get(name):
        del group[name]
    group.create_dataset(name, data=data)


def find_series(name, obj, sn):
    target_group_name = 'series_{}'.format(str(sn).zfill(3))
    if target_group_name in name:
        return obj


def loadResponses(ID, response_set_name='glom', get_voxel_responses=False):

    response_data = {}
    with h5py.File(ID.file_path, 'r') as experiment_file:
        find_partial = functools.partial(imaging_data.find_series, series_number=ID.series_number)
        roi_parent_group = experiment_file.visititems(find_partial)['aligned_response']
        roi_set_group = roi_parent_group[response_set_name]
        response_data['response'] = roi_set_group.get("response")[:]
        response_data['mask'] = roi_set_group.get("mask")[:]
        response_data['meanbrain'] = roi_set_group.get("meanbrain")[:]

        if get_voxel_responses:
            mask_vals = np.unique(response_data['mask'])[1:].astype(int)  # exclude first (0)
            voxel_responses = {}
            voxel_epoch_responses = {}
            for mv in mask_vals:
                voxel_responses[mv] = roi_set_group.get('voxel_resp_{}'.format(mv))[:].astype('float32')
                _, response_matrix = ID.getTrialAlignedVoxelResponses(voxel_responses[mv], dff=False)
                voxel_epoch_responses[mv] = response_matrix

    if get_voxel_responses:
        response_data['voxel_responses'] = voxel_responses
        response_data['voxel_epoch_responses'] = voxel_epoch_responses

    # epoch_response matrix for glom responses
    time_vector, response_matrix = ID.getTrialAlignedVoxelResponses(response_data.get('response'), dff=True)

    response_data['epoch_response'] = response_matrix
    response_data['time_vector'] = time_vector

    return response_data


def getGlomMaskDecoder(mask, base_dir='/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'):
    # Load mask key for VPN types
    vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

    vals = np.unique(mask)[1:]  # exclude first val (=0, not a glom)

    names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']
    return vals, names
