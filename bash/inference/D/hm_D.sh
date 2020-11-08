#!/bin/bash

# Del this file?

# Allows for not having to copy the models to vilio/data
loadfin=${1:-./data/LASTtrain.pth}
loadfin2=${2:-./data/LASTtraindev.pth}


# 50 Feats, Seed 49
cp ./data/hm_vgattr5050.tsv ./data/HM_img.tsv

python hm.py --seed 49 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin --exp D50 --subtest

python hm.py --seed 49 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 50 --loadfin $loadfin2 --exp D50 --subtest --combine

# 72 Feats, Seed 98
cp ./data/hm_vgattr7272.tsv ./data/HM_img.tsv

python hm.py --seed 98 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 72 --loadfin $loadfin --exp D72 --subtest

python hm.py --seed 98 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 72 --loadfin $loadfin2 --exp D72 --subtest --combine

# 36 Feats, Seed 147
cp ./data/hm_vgattr3636.tsv ./data/HM_img.tsv

python hm.py --seed 147 --model D \
--test dev_seen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin --exp D36 --subtest

python hm.py --seed 147 --model D \
--test test_seen,test_unseen --lr 1e-5 --batchSize 8 --tr bert-base-uncased --epochs 5 --tsv \
--num_features 36 --loadfin $loadfin2 --exp D36 --subtest --combine

# Simple Average
python utils/ens.py --enspath ./data/ --enstype sa --exp D365072