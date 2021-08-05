#!/bin/bash

# Paths
ROOT_DIR="/home/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"
ERNIE_ENV_ROOT_DIR="/home/ojrb2/ernie-vil"

# Parameters
EXPERIMENTS=('EL1' 'EL1c' 'EL5' 'EL5c' 'EL10' 'EL10c' 'EL15' 'EL15c' 'EL20' 'EL20c' 'EL36' 'EL36c' 'EL50' 'EL50c' 'ELV1' 'ELV1c' 'ELV5' 'ELV5c' 'ELV10' 'ELV10c' 'ELV15' 'ELV15c' 'ELV20' 'ELV20c' 'ELV36' 'ELV36c' 'ELV50' 'ELV50c' 'ES1' 'ES1c' 'ES5' 'ES5c' 'ES10' 'ES10c' 'ES15' 'ES15c' 'ES20' 'ES20c' 'ES36' 'ES36c' 'ES50' 'ES50c' 'ESV1' 'ESV1c' 'ESV5' 'ESV5c' 'ESV10' 'ESV10c' 'ESV15' 'ESV15c' 'ESV20' 'ESV20c' 'ESV36' 'ESV36c' 'ESV50' 'ESV50c' )
SEED=98
TOPK=-1  # Allows for quick test runs - Set topk to e.g. 20 
MODE=entity

# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z]+)([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [ $MODEL == 'ES' ] || [ $MODEL == 'EL' ] || [ $MODEL == 'ELV' ] || [ $MODEL == 'ESV' ]
  then
    sbatch \
      --output=outputs/$EXP \
      --error=outputs/$EXP \
      --export=ERNIE_ENV_ROOT_DIR=$ERNIE_ENV_ROOT_DIR,ROOT_DIR=$ROOT_DIR,EXP=$EXP \
      --job-name=$EXP \
      models/$MODEL.sh
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

