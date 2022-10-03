#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

# Params: (1) base directory, (2) series base name (no suffixes) (3) optional - type_of_transform (4) optional - meanbrain_n_frames (5) optional - aff_metric
# USAGE: sbatch moco_brainsss.sh /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/.../ TSeries-2022MMDD-00n

date

directory=$1
series_base=$2

brain_master="${series_base}_channel_1.nii"

echo $directory
echo $brain_master

# Optional params
type_of_transform="${3:-"Rigid"}"
meanbrain_n_frames="${4:-"100"}"
aff_metric="${5:-"mattes"}"  # also GC

output_format="nii"
echo $meanbrain_n_frames
echo $type_of_transform
echo $output_format

moco_directory="${directory}/moco/"

ml python/3.6 py-ants/0.3.2_py36

args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\",\"aff_metric\":\"$aff_metric\"}"

python3 -u /home/users/mhturner/brainsss/scripts/motion_correction.py $args
python3 -u /home/users/mhturner/glom_pop/moco/process_moco_channels.py ${moco_directory}${series_base} -one_channel
