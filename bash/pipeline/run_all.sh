#!/bin/bash
#$ -S /bin/bash

# Paths
ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio"
CONDA_ROOT_DIR="/home/miproj/4thyr.oct2020/ojrb2/miniconda3"


# Parameters
EXPERIMENTS=('U36a')
#seeds=(129)
TOPK=20  # Allows for quick test runs - Set topk to e.g. 20 & midsave to 5
SEED=43


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


# Run Models
source $CONDA_ROOT_DIR/bin/activate vilio
cd $ROOT_DIR
for EXP in "${EXPERIMENTS[@]}"; do
  read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z])([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"
  if [[ $MODEL == "U" ]]; then
    # Train Model
    python hm.py \
               --seed $SEED \
               --model U \
               --train train \
               --valid dev_seen \
               --test dev_seen \
               --lr 1e-5 \
               --batchSize 8 \
               --tr bert-large-cased \
               --epochs 5 \
               --tsv \
               --num_features $NUM_FEATS \
               --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
               --loadpre $MODEL_DIR/uniter-large.pt \
               --anno_dir $ANNO_DIR \
               --num_pos 6 \
               --contrib \
               --exp $EXP \
               --topk $TOPK \


    # Inference
    python hm.py \
               --seed $SEED \
               --model U \
               --train traindev \
               --valid dev_seen \
               --test test_seen,test_unseen \
               --lr 1e-5 \
               --batchSize 8 \
               --tr bert-large-cased \
               --epochs 5 \
               --tsv \
               --num_features $NUM_FEATS \
               --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
               --loadpre $MODEL_DIR/uniter-large.pt \
               --anno_dir $ANNO_DIR \
               --num_pos 6 \
               --contrib \
               --exp $EXP \
               --topk $TOPK

  elif [[ $MODEL == "D" ]]; then
    # Train Model
    python hm.py \
               --seed $SEED \
               --model D \
               --train train \
               --valid dev_seen \
               --test dev_seen \
               --lr 1e-5 \
               --batchSize 8 \
               --tr bert-base-uncased \
               --epochs 5 \
               --tsv \
               --num_features $NUM_FEATS \
               --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
               --loadpre $MODEL_DIR/devlbert.pth \
               --anno_dir $ANNO_DIR \
               --contrib \
               --exp $EXP \
               --topk $TOPK \

    # Inference
    python hm.py \
               --seed $SEED \
               --model D \
               --train traindev \
               --valid dev_seen \
               --test test_seen,test_unseen \
               --lr 1e-5 \
               --batchSize 8 \
               --tr bert-base-uncased \
               --epochs 5 \
               --tsv \
               --num_features $NUM_FEATS \
               --features $FEATURE_DIR/tsv/"$NUM_FEATS""$FLAGS".tsv \
               --loadpre $MODEL_DIR/devlbert.pth \
               --anno_dir $ANNO_DIR \
               --contrib \
               --exp $EXP \
               --topk $TOPK

  elif [[ $MODEL == "O" ]]; then
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
               --topk $TOPK

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
               --loadpre $DATA_DIR/LAST_$EXP.pth \
               --anno_dir $ANNO_DIR \
               --contrib \
               --exp $EXP \
               --topk $TOPK

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
               --loadpre $DATA_DIR/LAST_$EXP.pth \
               --anno_dir $ANNO_DIR \
               --contrib \
               --exp $EXP \
               --topk $TOPK

  elif [[ $MODEL == "X" ]]; then
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
  fi
done
