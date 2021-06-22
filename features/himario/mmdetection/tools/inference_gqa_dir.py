import os
import glob
import json
import mmcv
import numpy as np

from loguru import logger
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.visual_genome import GQA_CLASSES

config_file = 'configs/res2net/faster_rcnn_r2_101_fpn_2x_gqa.py'
checkpoint_file = '/Disk2/faster_rcnn_r2_101_fpn_2x_gqa_0922/epoch_2.pth'

img_dir = '/Disk2/hateful_memes/img_clean'
anno_out = '/Disk2/hateful_memes/clean_img_boxes_gqa.json'
imgs = glob.glob(os.path.join(img_dir, '*.png'))
# os.makedirs(out_dir, exist_ok=True)

print(F"Find {len(imgs)} imgs!")
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
split_img_counter = 0

with logger.catch():
    annotation = []
    for imid, img in enumerate(imgs):
        # test a single image and show the results
        # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
        bbox, aatr = inference_detector(model, img)
        detection = [
            {
                'class_id': cid,
                'class_name': GQA_CLASSES[cid],
                'box': box.tolist(),
                'xmin': float(box[0]),
                'ymin': float(box[1]),
                'xmax': float(box[2]),
                'ymax': float(box[3]),
                'score': float(box[4])
            }
            for cid, boxes in enumerate(bbox)
            if len(boxes) > 0
            for box in boxes
            if box[-1] > 0.2
        ]
        img_name = os.path.basename(img)
        annotation.append({
            'img_name': img_name,
            'boxes_and_score': detection,
        })
        print(imid, img, len(detection))

# visualize the results in a new window
# model.show_result(img, result)
with open(anno_out, 'w') as f:
    json.dump(annotation, f)