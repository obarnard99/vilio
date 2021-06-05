#!/bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"

# Parameters
EXPERIMENTS=('U5a' 'U10a' 'U15a' 'U20a' 'U36a' 'U50a' 'U72a' 'U5' 'U10' 'U15' 'U20' 'U36' 'U50' 'U72')
#seeds=(129)
#topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5

cd $ROOT_DIR/bash/pipeline

# Extract Features
qsub -l qp=cuda-low \
     -o outputs/feats \
     -e outputs/feats \
     -v EXPERIMENTS="${EXPERIMENTS[@]}" \
     -v ROOT_DIR=$ROOT_DIR \
     -v CONDA_ROOT_DIR=$CONDA_ROOT_DIR \
     feats.sh

# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [[ -e "$ROOT_DIR/data/features/tsv/$NUM_FEATS$FLAGS.tsv" ]]; then
    qsub -l qp=cuda-low \
         -o outputs/$EXP \
         -e outputs/$EXP \
         -v EXP=$EXP \
         -v ROOT_DIR=$ROOT_DIR \
         -v CONDA_ROOT_DIR=$CONDA_ROOT_DIR \
         -v TOPK=$TOPK \
         $MODEL.sh
    EXPERIMENTS=("${EXPERIMENTS[@]/$EXP}")
  fi
done
