#!/bin/bash

#SBATCH -A BYRNE-SL2-GPU
#SBATCH -p pascal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=05:00:00

. /etc/profile.d/modules.sh
module purge
module load rhel7/default-gpu
module load cuda/10.1
module load cudnn/7.6_cuda-10.1
source $VILIO_ROOT_DIR/bin/activate

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
