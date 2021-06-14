#!/bin/bash
DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data/features"
SPLIT=img_clean


python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vg \
--minboxes 36 --maxboxes 36 --dataroot $DATA_DIR

python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vg \
--minboxes 20 --maxboxes 20 --dataroot $DATA_DIR

# Attr feats:
python detectron2_mscoco_proposal_maxnms.py --batchsize 4 --split $SPLIT --weight vgattr \
--minboxes 20 --maxboxes 20 --dataroot $DATA_DIR
