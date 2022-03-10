#!/bin/bash
#
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_gp_1
#SBATCH --partition=trc
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/users/mhturner/glom_pop/moco/%x.%j.out
#SBATCH --open-mode=append

module use /home/groups/trc/modules
module load antspy/0.2.2

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-001
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-002
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-003

python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-009
python3 /home/users/mhturner/glom_pop/moco/moco.py /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/20220308/TSeries-20220308-010