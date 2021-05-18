#!/bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"

# Parameters
EXPERIMENTS=('U1' 'U1a' 'U1c' 'U1ac' 'U5' 'U5a' 'U5c' 'U5ac' 'U10' 'U10a' 'U10c' 'U10ac' 'U15' 'U15a' 'U15c' 'U15ac' 'U20' 'U20a' 'U20c' 'U20ac' 'U36' 'U36a' 'U36c' 'U36ac' 'U50' 'U50a' 'U50c' 'U50ac' 'U72' 'U72a' 'O1' 'O1a' 'O1c' 'O1ac' 'O5' 'O5a' 'O5c' 'O5ac' 'O10' 'O10a' 'O10c' 'O10ac' 'O15' 'O15a' 'O15c' 'O15ac' 'O20' 'O20a' 'O20c' 'O20ac' 'O36' 'O36a' 'O36c' 'O36ac' 'O50' 'O50a' 'O50c' 'O50ac' 'O72' 'O72a' 'D1' 'D1a' 'D1c' 'D1ac' 'D5' 'D5a' 'D5c' 'D5ac' 'D10' 'D10a' 'D10c' 'D10ac' 'D15' 'D15a' 'D15c' 'D15ac' 'D20' 'D20a' 'D20c' 'D20ac' 'D36' 'D36a' 'D36c' 'D36ac' 'D50' 'D50a' 'D50c' 'D50ac' 'D72' 'D72a' 'X1' 'X1a' 'X1c' 'X1ac' 'X5' 'X5a' 'X5c' 'X5ac' 'X10' 'X10a' 'X10c' 'X10ac' 'X15' 'X15a' 'X15c' 'X15ac' 'X20' 'X20a' 'X20c' 'X20ac' 'X36' 'X36a' 'X36c' 'X36ac' 'X50' 'X50a' 'X50c' 'X50ac' 'X72' 'X72a')
SEED=61
TOPK=-1  # Allows for quick test runs - Set topk to e.g. 20 
MODE=valid

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
       models/$MODE/$MODEL.sh
done

