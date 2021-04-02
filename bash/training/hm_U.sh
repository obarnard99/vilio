#!/bin/bash

# Allows for quick test runs - Set topk to e.g. 20 & midsave to 5
topk=-1


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
           --features ./data/features/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U50c \
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
           --features ./data/features/tsv_clean/hm_vgattr5050_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U50c \
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
           --features ./data/features/tsv_clean/hm_vgattr7272_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U72c \
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
           --features ./data/features/tsv_clean/hm_vgattr7272_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U72c \
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
           --features ./data/features/tsv_clean/hm_vgattr3636_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U36c \
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
           --features ./data/features/tsv_clean/hm_vgattr3636_clean.tsv \
           --loadpre ./data/models/uniter-large.pt \
           --num_pos 6 \
           --contrib \
           --exp U36c \
           --topk $topk \

