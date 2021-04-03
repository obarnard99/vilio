#!/bin/bash

# Constants
topk=-1  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
ANNO_DIR="$FEATURE_DIR/annotations"

# 50 Feats, Seed 43
python hm.py \
           --seed 43 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U50ac \
           --topk $topk \

python hm.py \
           --seed 43 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 50 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U50ac \
           --topk $topk \

# 72 Feats, Seed 86
python hm.py \
           --seed 86 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 72 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr7272_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U72ac \
           --topk $topk \

python hm.py \
           --seed 86 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 72 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr7272_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U72ac \
           --topk $topk \


# 36 Feats, Seed 129
python hm.py \
           --seed 129 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr3636_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U36ac \
           --topk $topk \

python hm.py \
           --seed 129 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 36 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr3636_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U36ac \
           --topk $topk \


# 5 Feats, Seed 11
python hm.py \
           --seed 11 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 5 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr55_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U5ac \
           --topk $topk \

python hm.py \
           --seed 11 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 5 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr55_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U5ac \
           --topk $topk \


# 10 Feats, Seed 97
python hm.py \
           --seed 97 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 10 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr1010_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U10ac \
           --topk $topk \

python hm.py \
           --seed 97 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 10 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr1010_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U10ac \
           --topk $topk \


# 15 Feats, Seed 142
python hm.py \
           --seed 142 \
           --model U \
           --train train \
           --valid dev_seen \
           --test dev_seen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 15 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr1515_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U15ac \
           --topk $topk \

python hm.py \
           --seed 142 \
           --model U \
           --train traindev \
           --valid dev_seen \
           --test test_seen,test_unseen \
           --lr 1e-5 \
           --batchSize 8 \
           --tr bert-large-cased \
           --epochs 5 \
           --tsv \
           --num_features 15 \
           --features $FEATURE_DIR/tsv_clean/hm_vgattr1515_clean.tsv \
           --loadpre $MODEL_DIR/uniter-large.pt \
           --anno_dir $ANNO_DIR \
           --num_pos 6 \
           --contrib \
           --exp U15ac \
           --topk $topk \
