#!/bin/bash

# Paths
ROOT_DIR="/home/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"
ERNIE_ENV_ROOT_DIR="/home/ojrb2/ernie-vil"
VILIO_ROOT_DIR="/home/ojrb2/vilio-env"

# Parameters
EXPERIMENTS=( 'O10' )
SEED=98
TOPK=-1  # Allows for quick test runs - Set topk to e.g. 20 
MODE=hpc

# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z]+)([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [ $MODE == 'hpc' ]
  then
    sbatch \
      --output=outputs/$EXP \
      --error=outputs/$EXP \
      --export=ERNIE_ENV_ROOT_DIR=$ERNIE_ENV_ROOT_DIR,ROOT_DIR=$ROOT_DIR,EXP=$EXP,VILIO_ROOT_DIR=$VILIO_ROOT_DIR,SEED=$SEED,TOPK=$TOPK \
      --job-name=$EXP \
      models/hpc/$MODEL.sh
  else
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
  fi
done

