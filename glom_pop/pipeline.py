import ants
import datetime
import nibabel as nib
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt

from glom_pop import dataio
from visanalysis.plugin import bruker
from visanalysis.util import h5io


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
    fig_directory = os.path.join(pipeline_dir, 'alignment_qc')
    # (1) ANTS OVERLAY
    series_name = os.path.split(brain_filepath)[-2]
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

    regions_fig, ax = plt.subplots(2, len(boxes)+1, figsize=(6, 2))
    [x.set_axis_off() for x in ax.ravel()]
    [x.set_xticks([]) for x in ax.ravel()]
    [x.set_yticks([]) for x in ax.ravel()]
    [x.axhline(dy/2, color='w') for x in ax.ravel()]
    [x.axvline(dx/2, color='w') for x in ax.ravel()]
    for b_ind, box in enumerate(boxes):
        z = zs[b_ind]

        ax[0, b_ind].imshow(ants.split_channels(meanbrain)[0][box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap='binary_r')
        ax[0, b_ind].set_title('Mean', fontsize=11, fontweight='bold')

        ind_red = ants.split_channels(ants.image_read(brain_filepath))[0]
        ax[1, b_ind].imshow(ind_red[box[0]:box[0]+dx, box[1]:box[1]+dy, z].T, cmap='binary_r')
        ax[1, b_ind].set_title(series_name, fontsize=11)

    fig_fp = os.path.join(fig_directory, '{}_regions.png'.format(series_name))
    regions_fig.savefig(fig_fp)
    print('Saved reg fig to {}'.format(fig_fp))



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



# def process_imports(path_to_import_folder,
#                     data_directory='/oak/stanford/groups/trc/data/Max/ImagingData/',
#                     datafile_dir='/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync/datafiles'):
#     """
#
#     args:
#     path_to_import_folder: path to imported data folder
#     data_directory: path to data, i.e. where to move new date dir
#     datafile_dir: path to directory where h5 datafiles should be copied
#
#     """
#
#     # (1) COPY TO NEW DATE DIRECTORY
#     output_subdir = os.path.split(path_to_import_folder)[-1]
#     # folder should be format YYYYMMDD
#     assert len(output_subdir) == 8
#     assert datetime.datetime.strptime(output_subdir, '%Y%m%d')
#
#     new_imaging_directory = os.path.join(data_directory, 'Bruker', output_subdir)
#     Path(new_imaging_directory).mkdir(exist_ok=True)  # make new directory for this date
#     print('Made directory {}'.format(new_imaging_directory))
#
#     for subdir in os.listdir(path_to_import_folder):  # one subdirectory per series
#         current_timeseries_directory = os.path.join(path_to_import_folder, subdir)
#         for fn in glob.glob(os.path.join(current_timeseries_directory, 'T*')):  # T series
#             dest = os.path.join(new_imaging_directory, os.path.split(fn)[-1])
#             shutil.copyfile(fn, dest)
#
#         for fn in glob.glob(os.path.join(current_timeseries_directory, 'Z*')):  # Z series
#             dest = os.path.join(new_imaging_directory, os.path.split(fn)[-1])
#             shutil.copyfile(fn, dest)
#
#     # (2) ATTACH VISPROTOCOL DATA
#     # Make a backup of raw visprotocol datafile before attaching data to it
#     experiment_file_name = '{}-{}-{}.hdf5'.format(output_subdir[0:4], output_subdir[4:6], output_subdir[6:8])
#     experiment_filepath = os.path.join(datafile_dir, experiment_file_name)
#     shutil.copy(experiment_filepath, os.path.join(data_directory, 'RawDataFiles', experiment_file_name))
#
#     plug = bruker.BrukerPlugin()
#     plug.attachData(experiment_file_name.split('.')[0], experiment_filepath, new_imaging_directory)
#
#     # Add analysis flags to each series
#     for sn in plug.getSeriesNumbers(file_path=experiment_filepath):
#         h5io.updateSeriesAttribute(file_path=experiment_filepath,
#                                    series_number=sn,
#                                    attr_key='include_in_analysis',
#                                    attr_val=True)
#
#         h5io.updateSeriesAttribute(file_path=experiment_filepath,
#                                    series_number=sn,
#                                    attr_key='anatomical_brain',
#                                    attr_val='')
#
#         h5io.updateSeriesAttribute(file_path=experiment_filepath,
#                                    series_number=sn,
#                                    attr_key='series_notes',
#                                    attr_val='')
#
#     print('Attached data to {}'.format(experiment_filepath))
