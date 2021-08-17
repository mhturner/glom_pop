#!/bin/bash
#
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=glom_resp
#SBATCH --partition=trc
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/users/mhturner/turner_analysis/job_outputs/%x.%j.out
#SBATCH --open-mode=append

# module load antspy/0.2.2


DATA_DIR=$OAK/data/Max/ImagingData/Bruker

for FN in "20210804-001 20210804-003" "20210804-002 20210804-003" "20210804-004 20210804-006"; do
  set -- $FN
  arrIN=(${1//-/ })
  DATE=${arrIN[0]}

  # STAGE DATAFILES TO SCRATCH
  cp $DATA_DIR/$DATE/TSeries-$1_reg.nii $SCRATCH/
  cp $DATA_DIR/$DATE/TSeries-$1.xml $SCRATCH/

  cp $DATA_DIR/$DATE/TSeries-$2_anatomical.nii $SCRATCH/
  cp $DATA_DIR/$DATE/TSeries-$2.xml $SCRATCH/

  python3 $HOME/glom_pop/scripts/attach_glom_responses.py $SCRATCH/TSeries-$1 $SCRATCH/TSeries-$2
done
