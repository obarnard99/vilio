#!/bin/bash
#$ -S /bin/bash

# Parameters
# EXP, ROOT_DIR and CONDA_ROOT_DIR passed as environment variables
read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
DATA_DIR="$ROOT_DIR/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"
SEED=43
TOPK=-1


source $CONDA_ROOT_DIR/bin/activate vilio
cd $ROOT_DIR


# Pretrain Model
python pretrain_bertX.py \
           --seed $SEED \
           --taskMaskLM \
           --wordMaskRate 0.15 \
           --train pretrain \
           --tsv \
           --llayers 12 \
           --rlayers 2 \
           --xlayers 5 \
           --batchSize 16 \
           --lr 0.5e-5 \
           --epochs 8 \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $MODEL_DIR/Epoch18_LXRT.pth \
           --anno_dir $ANNO_DIR \
           --topk $TOPK

# Train Model
python hm.py \
           --seed $SEED \
           --model X \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-base-uncased \
           --epochs 5 \
           --tsv \
           --llayers 12 \
           --rlayers 2 \
           --xlayers 5 \
           --num_features $NUM_FEATS \
           --loadpre $DATA_DIR/LAST_$EXP.pth \
           --anno_dir $ANNO_DIR \
           --swa \
           --exp $EXP \
           --topk $TOPK

# Inference
python hm.py \
           --seed $SEED \
           --model X \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-base-uncased \
           --epochs 5 \
           --tsv \
           --llayers 12 \
           --rlayers 2 \
           --xlayers 5 \
           --num_features $NUM_FEATS \
           --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
           --loadpre $DATA_DIR/LAST_$EXP.pth \
           --anno_dir $ANNO_DIR \
           --swa \
           --exp $EXP \
           --topk $TOPK
