#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

# Params: (1) base directory, (2) series base name (no suffixes)
# USAGE: sbatch moco_brainsss.sh /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/.../ TSeries-2022MMDD-00n

date

directory=$1
series_base=$2

brain_master="${series_base}_channel_1.nii"
brain_mirror="${series_base}_channel_2.nii"

echo $directory
echo $brain_master
echo $brain_mirror

# Optional params
type_of_transform="${3:-"Rigid"}"
output_format="nii"
echo $type_of_transform
echo $output_format

moco_directory="${directory}/moco/"

ml python/3.6 antspy/0.2.2

args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\",\"brain_mirror\":\"$brain_mirror\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\"}"

python3 -u /home/users/mhturner/brainsss/scripts/motion_correction.py $args
python3 -u /home/users/mhturner/glom_pop/moco/merge_channels.py ${moco_directory}${series_base}
