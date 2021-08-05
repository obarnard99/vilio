#!/bin/bash

#SBATCH -A BYRNE-SL2-GPU
#SBATCH -p pascal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00

. /etc/profile.d/modules.sh
module purge
module load rhel7/default-gpu
module load cuda/10.0
module load cudnn/7.6_cuda-10.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ERNIE_ENV_ROOT_DIR/TensorRT-6.0.1.5/lib
source $ERNIE_ENV_ROOT_DIR/bin/activate

read MODEL NUM_FEATS FLAGS <<< "$(sed -r 's/^([A-Z]*)([0-9]+)([a-z]*)/\1 \2 \3 /' <<< $EXP)"

cd $ROOT_DIR/ernie-vil

bash run_finetuning.sh hm \
  conf/hm/model_conf_hm \
  ./data/erniesmallvcr/vocab.txt \
  ./data/erniesmallvcr/ernie_vil.base.json \
  ./data/erniesmallvcr/params \
  train \
  2500 \
  $NUM_FEATS$FLAGS \
  $EXP

bash run_inference.sh hm \
  $NUM_FEATS$FLAGS \
  val \
  conf/hm/model_conf_hm \
  ./data/erniesmallvcr/vocab.txt \
  ./data/erniesmallvcr/ernie_vil.base.json \
  ./data/hm/models/$EXP \
  ./data/log \
  dev_all \
  $EXP \
  False

bash run_inference.sh hm \
  $NUM_FEATS$FLAGS \
  val \
  conf/hm/model_conf_hm \
  ./data/erniesmallvcr/vocab.txt \
  ./data/erniesmallvcr/ernie_vil.base.json \
  ./data/hm/models/$EXP \
  ./data/log \
  test_seen \
  $EXP \
  False

bash run_inference.sh hm \
  $NUM_FEATS$FLAGS \
  val \
  conf/hm/model_conf_hm \
  ./data/erniesmallvcr/vocab.txt \
  ./data/erniesmallvcr/ernie_vil.base.json \
  ./data/hm/models/$EXP \
  ./data/log \
  test_unseen \
  $EXP \
  False
