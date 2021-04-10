#!/bin/bash
#$ -S /bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"


# Parameters
EXPERIMENTS=('U36a')
#seeds=(129)
#topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5


# Extract Features
source $CONDA_ROOT_DIR/bin/activate detectron2
cd $ROOT_DIR/features/vilio/py-bottom-up-attention
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [[ ! -e "$ROOT_DIR/data/features/tsv/$NUM_FEATS$FLAGS.tsv" ]]; then
    echo "Extracting feats for $EXP"
    if [[ $FLAGS == *"a"* ]]; then
      WEIGHT="vgattr"
    else
      WEIGHT="vg"
    fi
    if [[ $FLAGS == *"c"* ]]; then
      SPLIT="img_clean"
    else
      SPLIT="img"
    fi
    python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight $WEIGHT \
    --minboxes $NUM_FEATS --maxboxes $NUM_FEATS --dataroot $ROOT_DIR/data
  else
    echo "$NUM_FEATS$FLAGS.tsv already exists"
  fi
done


# Run Models
cd $ROOT_DIR/bash/pipeline
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [[ $MODEL == "U" ]]; then
    qsub -l qp=cuda-low -o outputs/$EXP -e outputs/$EXP -v $EXP -v $ROOT_DIR -v $CONDA_ROOT_DIR U.sh
  elif [[ $MODEL == "O" ]]; then
    qsub -l qp=cuda-low -o outputs/$EXP -e outputs/$EXP -v $EXP -v $ROOT_DIR -v $CONDA_ROOT_DIR O.sh
  fi
done
