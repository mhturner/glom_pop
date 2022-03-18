#!/bin/bash
#SBATCH --job-name=moco
#SBATCH --partition=trc
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

date

DIRECTORY=$1
SERIES_BASE=$2

BRAIN_MASTER="${SERIES_BASE}_channel_1.nii"
BRAIN_MIRROR="${SERIES_BASE}_channel_2.nii"

ml python/3.6 antspy/0.2.2

args="{\"directory\":\"$DIRECTORY\",\"brain_master\":\"$BRAIN_MASTER\",\"brain_mirror\":\"$BRAIN_MIRROR\"}"

python3 -u ./motion_correction.py $args
python3 -u /home/users/mhturner/glom_pop/moco/h5_to_nii.py "${DIRECTORY}/${SERIES_BASE}"