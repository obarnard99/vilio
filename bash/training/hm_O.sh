#!/bin/bash

# Constants
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"
topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5

# 50 Feats, Seed 126
python pretrain/pretrain_bertO.py \
           --seed 126 \
           --taskMaskLM \
           --taskMatched \
           --wordMaskRate 0.15 \
           --train pretrain \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre $MODEL_DIR/pytorch_model.bin \
           --anno_dir $ANNO_DIR \
           --tr bert-large-uncased \
           --batchSize 8 \
           --lr 0.25e-5 \
           --epochs 8 \
           --topk $topk

python hm.py \
           --seed 126 \
           --model O \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O50ac \
           --topk $topk

python hm.py \
           --seed 126 \
           --model O \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O50ac \
           --topk $topk


# 50 VG feats, Seed 84
python pretrain/pretrain_bertO.py \
           --seed 84 \
           --taskMaskLM \
           --taskMatched \
           --wordMaskRate 0.15 \
           --train pretrain \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vg5050_clean.tsv \
           --loadpre $MODEL_DIR/pytorch_model.bin \
           --anno_dir $ANNO_DIR \
           --tr bert-large-uncased \
           --batchSize 8 \
           --lr 0.25e-5 \
           --epochs 8 \
           --topk $topk

python hm.py \
           --seed 84 \
           --model O \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vg5050_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O50c \
           --topk $topk

python hm.py \
           --seed 84 \
           --model O \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vg5050_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O50c \
           --topk $topk

# 36 Feats, Seed 42
python pretrain/pretrain_bertO.py \
           --seed 42 \
           --taskMaskLM \
           --taskMatched \
           --wordMaskRate 0.15 \
           --train pretrain \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vg3636_clean.tsv \
           --loadpre $MODEL_DIR/pytorch_model.bin \
           --anno_dir $ANNO_DIR \
           --tr bert-large-uncased \
           --batchSize 8 \
           --lr 0.25e-5 \
           --epochs 8 \
           --topk $topk

python hm.py \
           --seed 42 \
           --model O \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vg3636_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O36c \
           --topk $topk

python hm.py \
           --seed 42 \
           --model O \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-uncased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vg3636_clean.tsv \
           --loadpre $MODEL_DIR/LAST_BO.pth \
           --anno_dir $ANNO_DIR \
           --contrib \
           --exp O36c \
           --topk $topk

