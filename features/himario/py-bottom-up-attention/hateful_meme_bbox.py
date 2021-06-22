import detectron2
from detectron2.utils.logger import setup_logger
# setup_logger()
# import some common libraries
import io
import os
import glob
import json

import cv2
import fire
import torch
import numpy as np
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog



# Load VG Classes
data_path = 'data/genome/1600-400-20'
vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())
        
MetadataCatalog.get("vg").thing_classes = vg_classes


cfg = get_cfg()
cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
# cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl"

def get_bbox_of_dataset(dataset_root, output_path):
    
    det_annos = []
    img_list = glob.glob(os.path.join(dataset_root, 'img_clean', '*.png'))
    predictor = DefaultPredictor(cfg)

    for i, img_path in enumerate(img_list):
        print(f"{i}/{len(img_list)}")
        im = cv2.imread(img_path)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # showarray(im_rgb)
        outputs = predictor(im)
        pred = outputs['instances'].to('cpu')
        h, w = pred.image_size
        
        pred_boxes = pred.pred_boxes.tensor.cpu()
        pred_boxes /= torch.Tensor([w, h, w, h])
        boxes = pred_boxes.tolist()
        classes = pred.pred_classes.tolist()
        scores = pred.scores.tolist()
        class_names = [vg_classes[i] for i in classes]

        boxes_score = zip(
            boxes,
            scores,
            class_names,
            classes,
        )
        boxes_score = [
            {
                'ymin': float(b[0]),
                'xmin': float(b[1]),
                'ymax': float(b[2]),
                'xmax': float(b[3]),
                'score': float(s),
                'class_name': c,
                'class_id': int(ci),
            }
            for b, s, c, ci in list(boxes_score)
            if s > 0.2
        ]
        img_name = os.path.basename(img_path)
        det_anno = {
            'img_name': img_name,
            'boxes_and_score': boxes_score 
        }
        det_annos.append(det_anno)
    
    with open(output_path, mode='w') as output:
        json.dump(det_annos, output)


if __name__ == "__main__":
    fire.Fire(get_bbox_of_dataset)