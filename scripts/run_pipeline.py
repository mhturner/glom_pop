import ants
import argparse
import os
import shutil
import sys
import time
import datetime

from glom_pop import pipeline, util

t0_overall = time.time()

parser = argparse.ArgumentParser(description='Brain volume pipeline. Post-motion correction.')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')

parser.add_argument('-anatomical', action='store_true',
                    help='Flag to pass if series is anatomy scan. Else treats it as fxnal.')

parser.add_argument('--sync_dir', type=str, default='/oak/stanford/groups/trc/data/Max/Analysis/glom_pop/sync',
                    const=1, nargs='?',
                    help='Path to sync directory, on Oak')
args = parser.parse_args()
pipeline_dir = os.path.join(args.sync_dir, 'pipeline')
series_name = os.path.split(args.file_base_path)[-1]

# SET UP LOGGING
logfile = open(os.path.join(pipeline_dir, 'log_{}.txt'.format(series_name)), 'w')
sys.stdout = util.Tee(sys.stdout, logfile)
print(datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S"))

print('series_name: {}'.format(series_name))
print('sync_dir: {}'.format(args.sync_dir))
print('pipeline_dir: {}'.format(pipeline_dir))


if args.anatomical:
    print('------PROCESSING ANATOMICAL SCAN------')
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
    print('------REGISTER BRAIN------')
    transform_dir = os.path.join(args.sync_dir, 'transforms', 'meanbrain_anatomical')

    # Load meanbrain
    meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')
    meanbrain = ants.image_read(os.path.join(args.sync_dir, 'mean_brain', meanbrain_fn))

    path_to_registered_brain = pipeline.register_brain_to_reference(brain_file_path=anatomical_brain_path,
                                                                    reference_brain=meanbrain,
                                                                    transform_dir=transform_dir,
                                                                    type_of_transform='SyN',
                                                                    flow_sigma=3,
                                                                    total_sigma=0,
                                                                    initial_transform=None,
                                                                    mask=None,
                                                                    do_bias_correction=False)
    print('------/REGISTER BRAIN/------')

    print('------REGISTRATION QC------')
    pipeline.save_alignment_fig(path_to_registered_brain, meanbrain, pipeline_dir)
    print('------/REGISTRATION QC/------')

else:
    print('------PROCESSING FUNCTIONAL SCAN------')
    pass
    # FUNCTIONAL BRAIN SERIES: GET GLOM RESPONSES & BEHAVIOR AND ATTACH TO H5 FILE

    # ALIGN TO FUNCTIONAL AND ATTACH GLOMERULUS RESPONSES

    # ATTACH BEHAVIOR DATA

print('DONE WITH PIPELINE FOR {} ({:.0f} sec)'.format(args.file_base_path, time.time()-t0_overall))
print('------------------------------------')

logfile.close()
