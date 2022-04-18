import ants
import argparse
import nibabel as nib
import numpy as np
import os
import shutil
import time

from glom_pop import pipeline, dataio
from visanalysis.plugin import bruker


t0_overall = time.time()

parser = argparse.ArgumentParser(description='Brain volume pipeline. Post-motion correction.')
parser.add_argument('file_base_path', type=str,
                    help='Path to file base path with no suffixes. E.g. /path/to/data/TSeries-20210611-001')
args = parser.parse_args()

# GET CONFIG SETTINGS
sync_dir = dataio.get_config_file()['sync_dir']
print('sync_dir: {}'.format(sync_dir))

# (1) MAKE ANATOMICAL BRAIN FROM MOTION CORRECTED BRAIN
save_meanbrain = pipeline.get_anatomical_brain(args.file_base_path)

print('Saving meanbrain shape: {}'.format(save_meanbrain.shape))
save_path = '{}_anatomical.nii'.format(args.file_base_path)
# Note saving as ANTs image here (32 bit)
ants.image_write(save_meanbrain, save_path)

# Copy to sync dir
anatomical_brain_path = os.path.join(sync_dir, 'anatomical_brains', os.path.split(save_path)[-1])
shutil.copy(save_path, anatomical_brain_path)
print('Copied anatomical brain to {}'.format(anatomical_brain_path))

# (2) FOR TWO CHANNEL ChAT DATA: ALIGN ANATOMICAL TO MEANBRAIN
transform_dir = os.path.join(sync_dir, 'transforms', 'meanbrain_anatomical')

# Load meanbrain
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20211217')
meanbrain = ants.image_read(os.path.join(sync_dir, 'mean_brain', meanbrain_fn))

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
pipeline.save_alignment_fig(path_to_registered_brain, meanbrain, fig_directory)
# (3) ALIGN TO FUNCTIONAL AND ATTACH GLOMERULUS RESPONSES


# (4) ATTACH BEHAVIOR DATA
