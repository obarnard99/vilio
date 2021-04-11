#!/bin/bash
#$ -S /bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"

# Parameters
EXPERIMENTS=('U5a' 'U10a' 'U15a' 'U20a' 'U36a' 'U50a' 'U72a' 'U5' 'U10' 'U15' 'U20' 'U36' 'U50' 'U72')

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

