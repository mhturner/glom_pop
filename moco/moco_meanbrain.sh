#!/bin/bash
#
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_mb
#SBATCH --partition=trc
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

module use /home/groups/trc/modules
module load antspy/0.2.2

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-008 --meanbrain-only
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-009 --meanbrain-only

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-012 --meanbrain-only
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-013 --meanbrain-only

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-016 --meanbrain-only
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-017 --meanbrain-only

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-020 --meanbrain-only
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220301/TSeries-20220301-021 --meanbrain-only
