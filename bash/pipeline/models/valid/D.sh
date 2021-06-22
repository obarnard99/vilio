#!/bin/bash
#$ -S /bin/bash

# Parameters
# EXP, ROOT_DIR, CONDA_ROOT_DIR, TOPK, SEED passed as environment variables
read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
DATA_DIR="$ROOT_DIR/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"

source $CONDA_ROOT_DIR/bin/activate vilio
cd $ROOT_DIR

echo ""
echo "--------------------------------------------------- START ---------------------------------------------------"


# Train Model
python hm.py \
           --seed $SEED \
           --model D \
           --train train \
           --valid dev_all \
           --test dev_all,test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-base-uncased \
           --epochs 5 \
           --tsv \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $MODEL_DIR/devlbert.bin \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp $EXP \
           --topk $TOPK
