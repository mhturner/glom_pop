"""
Data pipeline functions.

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import ants
import nibabel as nib
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from glom_pop import dataio, alignment
from visanalysis.plugin import bruker
from visanalysis.analysis import imaging_data
from visanalysis.util import h5io

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # ANATOMICAL BRAIN & MEANBRAIN ALIGNMENT  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def register_brain_to_reference(brain_filepath,
                                reference_brain,
                                transform_dir,
                                type_of_transform='SyN',
                                flow_sigma=3,
                                total_sigma=0,
                                initial_transform=None,
                                mask=None,
                                do_bias_correction=False):
    """
    Register brain to reference_brain.
        Saves registered brain image and transform file

    :brain_filepath: .nii filepath to brain to register (MOVING)
    :reference_brain: two-channel ANTs image to register each brain to (FIXED)
    :transform_dir: path to transform directory (i.e. where to save transforms and results).
        Will make a subdirectory based on series name within this directory.
    :type_of_transform: for ants.registration()
    """
    print('Registering brain {}'.format(brain_filepath))
    t0 = time.time()
    series_name = os.path.split(brain_filepath)[-1].split('_')[0]

    # Make save paths for transforms
    image_transform_dir = os.path.join(transform_dir, series_name)
    os.makedirs(image_transform_dir, exist_ok=True)
    os.makedirs(os.path.join(image_transform_dir, 'forward'), exist_ok=True)
    os.makedirs(os.path.join(image_transform_dir, 'inverse'), exist_ok=True)

    individual_brain = ants.image_read(brain_filepath)

    if do_bias_correction:
        fixed_red = ants.n4_bias_field_correction(ants.split_channels(reference_brain)[0])
        moving_red = ants.n4_bias_field_correction(ants.split_channels(individual_brain)[0])

    else:
        fixed_red = ants.split_channels(reference_brain)[0]
        moving_red = ants.split_channels(individual_brain)[0]

    reg = ants.registration(fixed=fixed_red,
                            moving=moving_red,
                            type_of_transform=type_of_transform,
                            flow_sigma=flow_sigma,
                            total_sigma=total_sigma,
                            mask=mask,
                            initial_transform=initial_transform)

    # Copy transforms from tmp to long-term save dir
    dataio.save_transforms(reg, image_transform_dir)

    # Apply transform to each channel
    transform_list = dataio.get_transform_list(image_transform_dir, direction='forward')

    red_reg = ants.apply_transforms(fixed=ants.split_channels(reference_brain)[0],
                                    moving=ants.split_channels(individual_brain)[0],
                                    transformlist=transform_list,
                                    interpolator='nearestNeighbor',
                                    defaultvalue=0)

    green_reg = ants.apply_transforms(fixed=ants.split_channels(reference_brain)[1],
                                      moving=ants.split_channels(individual_brain)[1],
                                      transformlist=transform_list,
                                      interpolator='nearestNeighbor',
                                      defaultvalue=0)

    # Merge channels and save registered anatomical scan
    merged = ants.merge_channels([red_reg, green_reg])
    path_to_registered_brain = os.path.join(image_transform_dir,  'meanbrain_reg.nii')
    ants.image_write(merged, path_to_registered_brain)

    # Merge and save an overlay to check the registration
    overlay = ants.merge_channels([ants.split_channels(reference_brain)[0], red_reg])
    save_path = os.path.join(image_transform_dir,  'overlay_reg.nii')
    ants.image_write(overlay, save_path)

    print('Computed and saved transforms to {} ({} sec)'.format(image_transform_dir, time.time()-t0))

    return path_to_registered_brain


def save_alignment_fig(brain_filepath, meanbrain, pipeline_dir):
    """

    :brain_filepath: path/to/aligned/brain.nii
    :meanbrain: 2 channel chat meanbrain (ants image)

    """
    fig_directory = os.path.join(pipeline_dir, 'anatomical_qc')
    # (1) ANTS OVERLAY
    series_name = os.path.split(os.path.dirname(brain_filepath))[-1]
    ind_red = ants.split_channels(ants.image_read(brain_filepath))[0]
    ants.plot(ants.split_channels(meanbrain)[0], ind_red,
              cmap='Reds', overlay_cmap='Greens', axis=2, reorient=False,
              title=series_name)
    ants_fig = plt.gcf()
    fig_fp = os.path.join(fig_directory, '{}_ants.png'.format(series_name))
    ants_fig.savefig(fig_fp)
    print('Saved reg fig to {}'.format(fig_fp))

    # Show glom mask over some target areas
    dx = 100
    dy = 60

    boxes = [(55, 10), (30, 110), (120, 5)]
    zs = [10, 39, 39]

    regions_fig, ax = plt.subplots(len(boxes), 2, figsize=(4, 4))
    [x.set_axis_off() for x in ax.ravel()]
    [x.set_xticks([]) for x in ax.ravel()]
    [x.set_yticks([]) for x in ax.ravel()]
    [x.axhline(dy/2, color='w') for x in ax.ravel()]
    [x.axvline(dx/2, color='w') for x in ax.ravel()]
    for b_ind, box in enumerate(boxes):
        z = zs[b_ind]

        ax[b_ind, 0].imshow(ants.split_channels(meanbrain)[0][box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap='binary_r')

        ind_red = ants.split_channels(ants.image_read(brain_filepath))[0]
        ax[b_ind, 1].imshow(ind_red[box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap='binary_r')

        if b_ind == 0:
            ax[b_ind, 0].set_title('Mean', fontsize=11)
            ax[b_ind, 1].set_title(series_name, fontsize=11)

    fig_fp = os.path.join(fig_directory, '{}_regions.png'.format(series_name))
    regions_fig.savefig(fig_fp)
    print('Saved reg fig to {}'.format(fig_fp))


def get_functional_brain(file_base_path, channel=0):
    """
    Return single channel (xyzt) brain from time series brain

    file_base_path: Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001

    """
    metadata = bruker.getMetaData(file_base_path)
    c_dim = metadata['image_dims'][-1]

    brain = np.asarray(nib.load(file_base_path+'_reg.nii').dataobj, dtype='uint16')
    print('Loaded brain from {} (shape {})'.format(file_base_path, brain.shape))

    if c_dim == 1:
        print('One channel data...')
        functional_brain = brain
    elif c_dim == 2:
        print('Two channel data...')
        functional_brain = brain[..., channel]
    else:
        raise Exception('{}: Unrecognized c_dim.'.formt(file_base_path))

    return functional_brain


def get_anatomical_brain(file_base_path):
    """
    Return anatomical (xyzc) brain from motion-corrected time series brain

    file_base_path: Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001

    """
    metadata = bruker.getMetaData(file_base_path)
    c_dim = metadata['image_dims'][-1]

    spacing = [float(metadata['micronsPerPixel_XAxis']),
               float(metadata['micronsPerPixel_YAxis']),
               float(metadata['micronsPerPixel_ZAxis'])]
    print('Loaded metadata from {}'.format(file_base_path))
    print('Spacing (um/pix): {}'.format(spacing))
    print('Shape (xyztc) = {}'.format(metadata['image_dims']))

    meanbrain = np.asarray(nib.load(file_base_path+'_reg.nii').dataobj, dtype='uint16').mean(axis=3)
    print('Made meanbrain from {}'.format(file_base_path+'_reg.nii'))
    print('Meanbrain shape: {}'.format(meanbrain.shape))

    if c_dim == 1:
        print('One channel data...')
        save_meanbrain = ants.from_numpy(meanbrain[:, :, :], spacing=spacing)
    elif c_dim == 2:
        print('Two channel data...')
        ch1 = ants.from_numpy(meanbrain[:, :, :, 0], spacing=spacing)
        ch2 = ants.from_numpy(meanbrain[:, :, :, 1], spacing=spacing)
        save_meanbrain = ants.merge_channels([ch1, ch2])
    else:
        raise Exception('{}: Unrecognized c_dim.'.formt(file_base_path))

    return save_meanbrain


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # GLOM ALIGNMENT / RESPONSES  # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_glom_response_fig(glom_responses, merged, series_name, pipeline_dir):
    fig_directory = os.path.join(pipeline_dir, 'functional_qc')

    ants.plot(ants.split_channels(merged)[0], ants.split_channels(merged)[-1],
              cmap='Reds', overlay_cmap='Blues', overlay_alpha=0.5,
              axis=2, reorient=False,
              title=series_name, slices=np.arange(0, 12))
    ants_fig = plt.gcf()
    fig_fp = os.path.join(fig_directory, '{}_overlay.png'.format(series_name))
    ants_fig.savefig(fig_fp)
    print('Saved overlay fig to {}'.format(fig_fp))

    fh, ax = plt.subplots(len(glom_responses), 1, figsize=(12, 12))
    for gr_ind, gr in enumerate(glom_responses):
        ax[gr_ind].plot(gr, 'k-')

    fig_fp = os.path.join(fig_directory, '{}_resp.png'.format(series_name))
    fh.savefig(fig_fp)
    print('Saved response fig to {}'.format(fig_fp))

# # # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # BEHAVIOR DATA  # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # #


def process_behavior(video_filepath,
                     experiment_filepath,
                     series_number,
                     crop_window_size=[100, 100]  # rows x cols
                     ):

    print('Attached tags to {}'.format(experiment_filepath))

    frame_size = dataio.get_frame_size(video_filepath)
    print('Behavior frame size = {}'.format(frame_size))
    # crop = np.array(frame_size) - np.array(crop_window_size)
    crop_L = frame_size[1]/5
    crop_R = frame_size[1]/5
    crop_T = frame_size[0] / 2
    crop_B = frame_size[0] / 5

    # cropping: Pixels to trim from ((T, B), (L, R), (RGB_start, RGB_end))
    video_results = dataio.get_ball_movement(video_filepath,
                                             cropping=((crop_T, crop_B),
                                                       (crop_L, crop_R),
                                                       (0, 0)),
                                             )

    # Get frame timing from trigger signal
    voltage_trace, sample_rate = h5io.readDataSet(experiment_filepath, series_number,
                                                  group_name='stimulus_timing',
                                                  dataset_name='frame_monitor')
    frame_triggers = voltage_trace[0, :]  # First voltage trace is trigger readout
    video_results['frame_times'] = dataio.get_video_timing(frame_triggers, sample_rate)

    dataio.attach_behavior_data(experiment_filepath,
                                series_number,
                                video_results)

    return video_results


def save_behavior_fig(video_results, series_name, pipeline_dir):
    """

    :video_results: dict output from dataio.get_ball_movement
    :frame_times: from dataio.get_video_timing
    :series_name: str
    :pipeline_dir: str

    """
    fig_directory = os.path.join(pipeline_dir, 'functional_qc')

    fh, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 4]})
    ax[0].imshow(video_results['cropped_frame'], cmap='Greys_r')
    # Show cropped ball and overall movement trace for QC
    tw_ax = ax[1].twinx()
    tw_ax.fill_between(video_results['frame_times'][:video_results['binary_behavior'].shape[0]],
                       video_results['binary_behavior'],
                       color='k', alpha=0.5)
    ax[1].axhline(video_results['binary_thresh'], color='r')
    ax[1].plot(video_results['frame_times'][:video_results['rmse'].shape[0]],
               video_results['rmse'],
               'b')
    ax[1].set_title(series_name)

    fig_fp = os.path.join(fig_directory, '{}_beh.png'.format(series_name))
    fh.savefig(fig_fp)
    print('Saved behavior fig to {}'.format(fig_fp))
