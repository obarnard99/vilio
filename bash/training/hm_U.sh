#!/bin/bash

# Constants
topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"

# 50 Feats, Seed 43
python hm.py \
           --seed 47 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vg5050_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U50c \
           --topk $topk \

python hm.py \
           --seed 47 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vg5050_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U50c \
           --topk $topk \

# 72 Feats, Seed 86
python hm.py \
           --seed 93 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vg3636_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U36c \
           --topk $topk \

python hm.py \
           --seed 93 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vg3636_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U36c \
           --topk $topk \


# 36 Feats, Seed 129
python hm.py \
           --seed 111 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 20 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr2020_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U20ac \
           --topk $topk \

python hm.py \
           --seed 111 \
           --model U \
           --train trainlarge \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 20 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr2020_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U20ac \
           --topk $topk \

