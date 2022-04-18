import ants
import argparse
import os
from pathlib import Path
import shutil
import time

from glom_pop import pipeline

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

# GET CONFIG SETTINGS
print('sync_dir: {}'.format(args.sync_dir))

if args.anatomical:
    # ANATOMICAL BRAIN SERIES: REGISTER TO MEANBRAIN

    # MAKE ANATOMICAL BRAIN FROM MOTION CORRECTED BRAIN
    save_meanbrain = pipeline.get_anatomical_brain(args.file_base_path)

    print('Saving meanbrain shape: {}'.format(save_meanbrain.shape))
    save_path = '{}_anatomical.nii'.format(args.file_base_path)
    # Note saving as ANTs image here (32 bit)
    ants.image_write(save_meanbrain, save_path)

    # Copy to sync dir
    anatomical_brain_path = os.path.join(args.sync_dir, 'anatomical_brains', os.path.split(save_path)[-1])
    shutil.copy(save_path, anatomical_brain_path)
    print('Copied anatomical brain to {}'.format(anatomical_brain_path))

    # FOR TWO CHANNEL ChAT DATA: ALIGN ANATOMICAL TO MEANBRAIN
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

    fig_directory = os.path.join(transform_dir, 'alignment_qc')
    Path(fig_directory).mkdir(exist_ok=True)  # make new directory for this date
    pipeline.save_alignment_fig(path_to_registered_brain, meanbrain, fig_directory)

else:
    pass
    # FUNCTIONAL BRAIN SERIES: GET GLOM RESPONSES & BEHAVIOR AND ATTACH TO H5 FILE

    # ALIGN TO FUNCTIONAL AND ATTACH GLOMERULUS RESPONSES

    # ATTACH BEHAVIOR DATA

print('DONE WITH PIPELINE FOR {} ({:.0f} SEC)'.format(args.file_base_path, time.time()-t0_overall))
