"""
Processing pipeline for single brains.

Before this:
1. process_imports
2. moco_brainsss
3. Tag functional series in H5 file with associated anatomical series name

Pipeline steps:
If anatomical scan (flag -anatomical):
1. Make anatomical brain from T series
2. Register anatomical brain to meanbrain and save transform data

If functional scan:
1. For ChAT brains: attach extracted glom responses to H5
2. For series with behavior videos: process behavioral video and attach results to H5

e.g. usage:
python3 run_pipeline.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-001 -chat

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import ants
import argparse
import datetime
import glob
import os
import shutil
import sys
import time

from glom_pop import pipeline, util
from visanalysis.util import h5io


t0_overall = time.time()

parser = argparse.ArgumentParser(description='Brain volume pipeline. Post-motion correction.')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')

parser.add_argument('-chat', action='store_true',
                    help='Flag to pass if series is a chat brain - align to meanbrain and extract glom responses for fxnal')

parser.add_argument('-anatomical', action='store_true',
                    help='Flag to pass if series is anatomy scan. Else treats it as fxnal.')

parser.add_argument('--sync_dir', type=str, default='/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync',
                    const=1, nargs='?',
                    help='Path to sync directory, on Oak')

parser.add_argument('--behavior_dir', type=str, default='/oak/stanford/groups/trc/data/Max/ImagingData/BrukerCamera',
                    const=1, nargs='?',
                    help='Path to behavior parent directory, on Oak')
args = parser.parse_args()
pipeline_dir = os.path.join(args.sync_dir, 'pipeline')
series_name = os.path.split(args.file_base_path)[-1]
date_str = series_name.split('-')[1]
series_number = int(series_name.split('-')[-1])

# SET UP LOGGING
logfile = open(os.path.join(pipeline_dir, 'logs', '{}.txt'.format(series_name)), 'w')
sys.stdout = util.Tee(sys.stdout, logfile)
print(datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S"))

print('series_name: {}'.format(series_name))
print('date_str: {}'.format(date_str))
print('series_number: {}'.format(series_number))
print('sync_dir: {}'.format(args.sync_dir))
print('behavior_dir: {}'.format(args.behavior_dir))
print('pipeline_dir: {}'.format(pipeline_dir))


if args.anatomical:
    print('**** ANATOMICAL SCAN ****')
    # ANATOMICAL BRAIN SERIES: REGISTER TO MEANBRAIN

    # MAKE ANATOMICAL BRAIN FROM MOTION CORRECTED BRAIN
    print('------MAKE ANATOMICAL BRAIN------')
    save_meanbrain = pipeline.get_anatomical_brain(args.file_base_path)

    print('Saving meanbrain shape: {}'.format(save_meanbrain.shape))
    print('Saving meanbrain channels: {}'.format(save_meanbrain.components))
    save_path = '{}_anatomical.nii'.format(args.file_base_path)
    # Note saving as ANTs image here (32 bit)
    ants.image_write(save_meanbrain, save_path)

    # Copy to sync dir
    anatomical_brain_path = os.path.join(args.sync_dir, 'anatomical_brains', os.path.split(save_path)[-1])
    shutil.copy(save_path, anatomical_brain_path)
    print('Copied anatomical brain to {}'.format(anatomical_brain_path))
    print('------/MAKE ANATOMICAL BRAIN/------')

    # FOR TWO CHANNEL ChAT DATA: ALIGN ANATOMICAL TO MEANBRAIN
    if args.chat:
        print('------REGISTER ChAT BRAIN TO MEANBRAIN------')
        transform_dir = os.path.join(args.sync_dir, 'transforms', 'meanbrain_anatomical')

        # Load meanbrain
        meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')
        meanbrain = ants.image_read(os.path.join(args.sync_dir, 'mean_brain', meanbrain_fn))

        # Register anatomical brain to meanbrain. Saves transformed image and transform files
        path_to_registered_brain = pipeline.register_brain_to_reference(brain_filepath=anatomical_brain_path,
                                                                        reference_brain=meanbrain,
                                                                        transform_dir=transform_dir,
                                                                        type_of_transform='SyN',
                                                                        flow_sigma=3,
                                                                        total_sigma=0,
                                                                        initial_transform=None,
                                                                        mask=None,
                                                                        do_bias_correction=False)

        pipeline.save_alignment_fig(path_to_registered_brain, meanbrain, pipeline_dir)
        print('------/REGISTER ChAT BRAIN TO MEANBRAIN/------')

else:  # Not anatomical - functional scan
    print('------PROCESSING FUNCTIONAL SCAN------')
    if args.chat:  # GET GLOM RESPONSES & ATTACH TO H5
        # Load h5 file
        datafile_dir = os.path.join(args.sync_dir, 'datafiles')
        experiment_file_name = '{}-{}-{}.hdf5'.format(date_str[0:4], date_str[4:6], date_str[6:8])
        experiment_filepath = os.path.join(datafile_dir, experiment_file_name)

        if h5io.seriesExists(experiment_filepath, series_number):
            glom_responses = pipeline.align_glom_responses(experiment_filepath,
                                                           series_number,
                                                           args.sync_dir,
                                                           meanbrain_datestr='20211217')
            pipeline.save_glom_response_fig(glom_responses,
                                            pipeline_dir)
        else:
            print('No existing series {} found in {}'.format(series_number, experiment_filepath))
            print('No glomeruli responses attached')
            # TODO: Make a new series group for series with no visual stimuli
            # How to figure out fly id?
            # h5io.createEpochRunGroup(experiment_filepath, fly_id, series_number)

    # ATTACH BEHAVIOR DATA
    series_dir = 'series' + str(series_number).zfill(3)
    video_filepaths = glob.glob(os.path.join(args.behavior_dir, date_str, series_dir) + "/*.avi")
    if len(video_filepaths) == 0:
        print('No behavior video found for {}: {}'.format(date_str, series_dir))
    elif len(video_filepaths) == 1:  # should be just one .avi in there
        print('------PROCESSING BEHAVIOR DATA------')
        # Process rms image difference on ball. Attach results to h5 file
        video_results = pipeline.process_behavior(video_filepaths[0],
                                                  experiment_filepath,
                                                  series_number,
                                                  crop_window_size=[100, 100])
        pipeline.save_behavior_fig(video_results,
                                   series_name,
                                   pipeline_dir)
        print('------/PROCESSING BEHAVIOR DATA/------')

    elif len(video_filepaths) > 1:
        print('More than one behavior video found in {}'.format(os.path.join(args.behavior_dir, date_str, series_dir)))


print('DONE WITH PIPELINE FOR {} ({:.0f} sec)'.format(args.file_base_path, time.time()-t0_overall))
print('------------------------------------')

logfile.close()
