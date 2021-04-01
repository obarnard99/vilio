#!/bin/bash
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data/features"
SPLIT=img_clean

# Attr feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 36 --maxboxes 36 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 50 --maxboxes 50 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 72 --maxboxes 72 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 10 --maxboxes 100 --dataroot $DATA_DIR

# VG feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vg \
--minboxes 50 --maxboxes 50 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vg \
--minboxes 10 --maxboxes 100 --dataroot $DATA_DIR