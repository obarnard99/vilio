#!/bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"

# Parameters
EXPERIMENTS=('U36' 'U50' 'U72' 'O5a' 'O10a' 'O15a' 'O20a' 'O36a' 'O50a' 'O72a' 'O5' 'O10' 'O15' 'O20' 'O36' 'O50' 'O72')
SEED=43
TOPK=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5

# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  qsub -l qp=cuda-low \
       -o outputs/$EXP \
       -e outputs/$EXP \
       -v EXP=$EXP \
       -v ROOT_DIR=$ROOT_DIR \
       -v CONDA_ROOT_DIR=$CONDA_ROOT_DIR \
       -v TOPK=$TOPK \
       -v SEED=$SEED \
       -N $EXP \
       models/$MODEL.sh
done

