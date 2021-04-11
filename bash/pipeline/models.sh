#!/bin/bash
#$ -S /bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"

# Parameters
EXPERIMENTS=('U36a')
#seeds=(129)
TOPK=20  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5

# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  qsub -l qp=cuda-low -o outputs/$EXP -e outputs/$EXP -v EXP=$EXP -v ROOT_DIR=$ROOT_DIR -v CONDA_ROOT_DIR=$CONDA_ROOT_DIR -v TOPK=$TOPK $MODEL.sh
done

