#!/bin/bash
#$ -S /bin/bash

# Parameters
EXP=$1
read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
ROOT_DIR=$2
CONDA_ROOT_DIR=$3
DATA_DIR="$ROOT_DIR/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"
SEED=43
topk=-1


source $CONDA_ROOT_DIR/bin/activate vilio
cd $ROOT_DIR


# Pretrain Model
python pretrain_bertO.py \
           --seed $SEED \
           --taskMaskLM \
           --taskMatched \
           --wordMaskRate 0.15 \
           --train pretrain \
           --tsv \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $MODEL_DIR/pytorch_model.bin \
           --anno_dir $ANNO_DIR \
           --tr bert-large-uncased \
           --batchSize 8 \
           --lr 0.25e-5 \
           --epochs 8 \
           --topk $topk


# Train Model
python hm.py \
           --seed $SEED \
           --model O \
           --train trainlarge \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $DATA_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp $EXP \
           --topk $topk


# Inference
python hm.py \
           --seed $SEED \
           --model O \
           --train trainlarge \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $DATA_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp $EXP \
           --topk $topk
