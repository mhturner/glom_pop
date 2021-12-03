#!/bin/bash
#
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_mb
#SBATCH --partition=trc
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

module use /home/groups/trc/modules
module load antspy/0.2.2

python3 /home/users/mhturner/glom_pop/moco/moco_meanbrain.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20211202/TSeries-20211202-002

python3 /home/users/mhturner/glom_pop/moco/moco_meanbrain.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20211202/TSeries-20211202-004

python3 /home/users/mhturner/glom_pop/moco/moco_meanbrain.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20211202/TSeries-20211202-006

python3 /home/users/mhturner/glom_pop/moco/moco_meanbrain.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20211202/TSeries-20211202-008
