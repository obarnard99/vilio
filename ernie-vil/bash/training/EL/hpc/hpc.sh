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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ojrb2/ernie-vil/TensorRT-6.0.1.5/lib
source ~/ernie-vil/bin/activate

### ATT 36

#mv ./data/hm/hm_vgattr10100.tsv ./data/hm/HM_gt_img.tsv
#mv ./data/hm/hm_vgattr3636.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm \
  conf/hm/model_conf_hm \
  ./data/ernielarge/vocab.txt \
  ./data/ernielarge/ernie_vil.large.json \
  ./data/ernielarge/params \
  train \
  2500

bash run_inference.sh hm \
  "" \
  val \
  conf/hm/model_conf_hm \
  ./data/ernielarge/vocab.txt \
  ./data/ernielarge/ernie_vil.large.json \
  ./output_hm/step_2500train \
  ./data/log \
  dev_all \
  EL10a \
  False

bash run_inference.sh hm \
  "" \
  val \
  conf/hm/model_conf_hm \
  ./data/ernielarge/vocab.txt \
  ./data/ernielarge/ernie_vil.large.json \
  ./output_hm/step_2500traindev \
  ./data/log \
  test_seen \
  EL10a \
  False

bash run_inference.sh hm \
  "" \
  val \
  conf/hm/model_conf_hm \
  ./data/ernielarge/vocab.txt \
  ./data/ernielarge/ernie_vil.large.json \
  ./output_hm/step_2500traindev \
  ./data/log \
  test_unseen \
  EL10a \
  False
