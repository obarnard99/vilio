#!/bin/bash

# Paths
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"
TSV_DIR="$FEATURE_DIR/tsv"


# Parameters
EXPERIMENTS=('U36ac')
#seeds=(129)
topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5


# Extract Features
for EXP in "${EXPERIMENTS[@]}"; do
  if [[ ! -e "$TSV_DIR/$EXP.tsv" ]]; then
    echo "Extracting feats for $EXP"
    read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
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
    --minboxes $NUM_FEATS --maxboxes $NUM_FEATS --dataroot $DATA_DIR
  else
    echo "$EXP.tsv already exists"
  fi
done
