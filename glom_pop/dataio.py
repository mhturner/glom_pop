"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import functools
import inspect
import os
import shutil
import glob

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import yaml
import xml.etree.ElementTree as ET
import pims
from sewar.full_ref import rmse as sewar_rmse
from scipy.signal import resample, savgol_filter
from scipy.interpolate import interp1d
from skimage import filters

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


def get_ft_datapath(ID, ft_dir):
    series_number = ID.series_number
    file_name = os.path.split(ID.file_path)[-1].split('.')[0]
    look_path = os.path.join(ft_dir,
                             file_name.replace('-', ''),
                             'series{}'.format(str(series_number).zfill(3)))
    glob_res = glob.glob(os.path.join(look_path, '*.dat'))
    if len(glob_res) == 0:
        return False
    elif len(glob_res) > 1:
        print('Warning! Multiple FT .dat files found at {}'.format(look_path))
        return glob_res[0]
    elif len(glob_res) == 1:
        return glob_res[0]



def load_fictrac_data(ID, ft_data_path, process_behavior=True, exclude_thresh=None, show_qc=True):
    # Imaging time stamps
    imaging_time_vector = ID.getResponseTiming()['time_vector']

    # exclude_thresh: deg per sec
    ft_data = pd.read_csv(ft_data_path, header=None)

    frame = ft_data.iloc[:, 0]
    timestamp = ft_data.iloc[:, 21].values / 1e3  # msec -> sec
    timestamp = timestamp - timestamp[0]
    fps = 1 / np.mean(np.diff(timestamp))

    xrot = ft_data.iloc[:, 5] * 180 / np.pi * fps  # rot  --> deg/sec
    yrot = ft_data.iloc[:, 6] * 180 / np.pi * fps  # rot  --> deg/sec
    zrot = ft_data.iloc[:, 7] * 180 / np.pi * fps  # rot  --> deg/sec

    xrot_filt = savgol_filter(xrot, 151, 3)
    yrot_filt = savgol_filter(yrot, 151, 3)
    zrot_filt = savgol_filter(zrot, 151, 3)

    walking_mag = np.sqrt(xrot_filt**2 + yrot_filt**2 + zrot_filt**2)

    # Downsample from camera frame rate to imaging frame rate
    walking_mag_interp = interp1d(timestamp, walking_mag, kind='linear', bounds_error=False, fill_value=np.nan)
    turning_interp = interp1d(timestamp, zrot_filt, kind='linear', bounds_error=False, fill_value=np.nan)
    speed_interp = interp1d(timestamp, yrot_filt, kind='linear', bounds_error=False, fill_value=np.nan)

    walking_mag_ds = walking_mag_interp(imaging_time_vector)
    turning_ds = turning_interp(imaging_time_vector)
    speed_ds = speed_interp(imaging_time_vector)

    if exclude_thresh is not None:
        # Filter to remove timepoints when tracking is lost
        turning_ds[walking_mag_ds > exclude_thresh] = 0
        speed_ds[walking_mag_ds > exclude_thresh] = 0
        walking_mag_ds[walking_mag_ds > exclude_thresh] = 0

    thresh = filters.threshold_li(walking_mag_ds)
    binary_behavior_ds = (walking_mag_ds > thresh).astype('int')
    _, behavior_binary_matrix = ID.getEpochResponseMatrix(binary_behavior_ds[np.newaxis, :],
                                                          dff=False)
    _, walking_response_matrix = ID.getEpochResponseMatrix(walking_mag_ds[np.newaxis, :],
                                                           dff=False)

    _, turning_response_matrix = ID.getEpochResponseMatrix(turning_ds[np.newaxis, :],
                                                           dff=False)

    _, speed_response_matrix = ID.getEpochResponseMatrix(speed_ds[np.newaxis, :],
                                                         dff=False)

    is_behaving = ID.getResponseAmplitude(behavior_binary_matrix, metric='mean') > 0.25
    walking_amp = ID.getResponseAmplitude(walking_response_matrix, metric='mean')
    turning_amp = ID.getResponseAmplitude(turning_response_matrix, metric='mean')
    speed_amp = ID.getResponseAmplitude(speed_response_matrix, metric='mean')

    behavior_data = {'walking_mag': walking_mag,  # n video frames
                     'walking_mag_ds': walking_mag_ds,  # n imaging frames
                     'behavior_binary_matrix': behavior_binary_matrix,  # 1 x trials x time
                     'walking_response_matrix': walking_response_matrix,  # 1 x trials x time
                     'turning_response_matrix': turning_response_matrix,  # 1 x trials x time
                     'speed_response_matrix': speed_response_matrix,  # 1 x trials x time
                     'walking_amp': walking_amp,  # n trials
                     'turning_amp': turning_amp,
                     'speed_amp': speed_amp,
                     'is_behaving': is_behaving,  # n trials
                     'thresh': thresh,
                     'timestamp': timestamp  # sec
                     }

    if show_qc:
        fh, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].plot(walking_mag_ds)
        ax[0].axhline(thresh, color='r')
        ax[1].hist(walking_mag_ds, 100)
        ax[1].axvline(thresh, color='r')
        ax[0].set_title('{}: {}'.format(os.path.split(ID.file_path)[-1], ID.series_number))
    return behavior_data


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


def filter_epoch_response_matrix(response_data, included_vals, glom_size_threshold=10):
    # epoch_response_matrix: shape=(gloms, trials, time)
    epoch_response_matrix = np.zeros((len(included_vals), response_data.get('epoch_response').shape[1], response_data.get('epoch_response').shape[2]))
    epoch_response_matrix[:] = np.nan

    for val_ind, included_val in enumerate(included_vals):
        new_glom_size = np.sum(response_data.get('mask') == included_val)

        if new_glom_size > glom_size_threshold:
            pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
            epoch_response_matrix[val_ind, :, :] = response_data.get('epoch_response')[pull_ind, :, :]
        else:  # Exclude because this glom, in this fly, is too tiny
            pass

    return epoch_response_matrix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  Interacting with behavior video  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_frame_size(filepath):
    frame = pims.as_grey(pims.Video(filepath))[0]
    return frame.shape


def get_video_timing(frame_triggers, sample_rate=10000):
    # Use frame trigger voltage output to find frame times in bruker time
    # shift & normalize so trace lives on [0 1]
    frame_triggers = frame_triggers - np.min(frame_triggers)
    frame_triggers = frame_triggers / np.max(frame_triggers)

    # find trigger up times
    threshold = 0.5
    V_orig = frame_triggers[0:-2]
    V_shift = frame_triggers[1:-1]
    frame_times = np.where(np.logical_and(V_orig < threshold, V_shift >= threshold))[0] + 1

    frame_times = frame_times / sample_rate  # Seconds

    print('{} frame triggers sent'.format(frame_times.shape[0]))

    return frame_times


def get_ball_movement(filepath,
                      cropping=((90, 0), (10, 20), (0, 0)),  # Pixels to trim from ((L, R), (T, B), (RGB_start, RGB_end))
                      ):
    # Load and crop vid as a pims object
    whole_vid = pims.as_grey(pims.Video(filepath))
    cropped_vid = pims.as_grey(pims.process.crop(pims.Video(filepath), cropping))

    # Measure ball movement by computing rmse between successive frames
    ball_rmse = np.array([sewar_rmse(cropped_vid[f], cropped_vid[f+1]) for f in range(len(cropped_vid)-1)])

    # Binarize using Otsu threshold
    # i.e. minimize within-class variance
    thresh = filters.threshold_otsu(ball_rmse)
    binary_behavior = ball_rmse > thresh

    print('{} frames in movie'.format(ball_rmse.shape[0]+1))

    video_results = {'frame': whole_vid[1],
                     'cropped_frame': cropped_vid[1],
                     'rmse': ball_rmse,
                     'binary_behavior': binary_behavior,
                     'binary_thresh': thresh
                     }

    return video_results


def attach_behavior_data(file_path,
                         series_number,
                         video_results):
    with h5py.File(file_path, 'r+') as experiment_file:
        find_partial = functools.partial(h5io.find_series, sn=series_number)
        epoch_run_group = experiment_file.visititems(find_partial)
        behavior_group = epoch_run_group.require_group('behavior')

        for key in video_results:
            overwrite_dataset(behavior_group, key, video_results[key])
