#!/bin/bash
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data/features"
SPLIT=img_clean

# Attr feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 5 --maxboxes 5 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 10 --maxboxes 10 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 15 --maxboxes 15 --dataroot $DATA_DIR