import functools
import os
import io
import json
import time
import random

# import some common libraries
import cv2
import fire
import torch
import PIL.Image
import numpy as np
from torch import nn
import detectron2
from loguru import logger
import imgaug

from albumentations import (
    BboxParams,
    Crop,
    Compose,
    ShiftScaleRotate,
    RandomBrightness,
    RandomContrast,
    RandomScale,
    Rotate,
    HorizontalFlip,
    MedianBlur,
)

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

NUM_OBJECTS = 18


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


"""
Load visual gnome labels
"""

# Load VG Classes
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'data/genome/1600-400-20')


vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())


MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

"""
Load Fater R-CNN
"""

cfg = get_cfg()
cfg.merge_from_file(os.path.join(current_dir, "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
# cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
cfg.MODEL.DEVICE = 'cuda'
predictor = DefaultPredictor(cfg)


def doit(raw_image, raw_boxes):
    # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes))
    
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        boxes.tensor = boxes.tensor.to(cfg.MODEL.DEVICE)
        
        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        # Predict classes        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled) and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores = max_attr_prob,
            attr_classes = max_attr_label
        )
        
        return instances, roi_features


def doit_without_boxes(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        
        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor    
        
        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
                
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label
        
        print(instances)
        
        return instances, roi_features


def get_box_feature(boxes, im):
    h, w, c = im.shape
    features = []
    for box in boxes:
        norm_x = box['xmin']
        norm_y = box['ymin']
        norm_w = box['xmax'] - box['xmin']
        norm_h = box['ymax'] - box['ymin']
        features.append([
            norm_x,
            norm_y,
            norm_x + norm_w,
            norm_y + norm_h,
            norm_w,
            norm_h,
        ])
    # (num_boxes, 6)
    return torch.Tensor(features)


def apply_augs(im, boxes):
    data_dict = {
        'image': im,
        'bboxes': [
            [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            for box in boxes
        ],
        'fake_label': [0] * len(boxes)
    }
    # print(data_dict['bboxes'])
    
    bbox_params = BboxParams(
        format='albumentations',
        min_area=0, 
        min_visibility=0.2,
        label_fields=['fake_label'])
    
    album_augs = [
        HorizontalFlip(p=0.5),
        # RandomBrightness(limit=0.3, p=0.5),
        # RandomContrast(limit=0.3, p=0.5),
        RandomScale(scale_limit=(-0.3, 0.0), p=0.3),
        # MedianBlur(blur_limit=5, p=0.3),
        # Rotate(limit=10, p=0.25),
    ]
    album_augs = Compose(album_augs, bbox_params=bbox_params)
    
    new_data_dict = album_augs(**data_dict)
    # print(new_data_dict['bboxes'])

    new_boxes = [
        {
            **boxes[i],
            'xmin': new_data_dict['bboxes'][i][0],
            'ymin': new_data_dict['bboxes'][i][1],
            'xmax': new_data_dict['bboxes'][i][2],
            'ymax': new_data_dict['bboxes'][i][3],
        }
        for i in range(len(boxes))
    ]
    return new_data_dict['image'], new_boxes


def oid_boxes(json_path, dataset_root, output_path, augment=False, random_seed=None):
    if random_seed is None:
        random_seed = int(time.time()) % 10000
    if augment:
        logger.info(f"random_seed: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        imgaug.random.seed(random_seed)
    
    with open(json_path, mode='r') as anno_file:
        box_anno = json.load(anno_file)
    
    new_annos = []
    name2feature = {}
    for i, img_anno in enumerate(box_anno):
        boxes = img_anno['boxes_and_score']
        img_name = img_anno['img_name']
        
        # NOTE: using super resolution image
        img_path = os.path.join(dataset_root, 'img_clean', img_name)
        if not os.path.exists(img_path) or True:
            img_path = os.path.join(dataset_root, 'img', img_name)
        logger.warning(f"Loading image from: {img_path}")
        im = cv2.imread(img_path)
        if augment:
            im, boxes = apply_augs(im, boxes)
            # print(f"[{i}]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print(boxes)
        h, w, c = im.shape

        if len(boxes) > 0:
            box_np = [[
                box['xmin'] * w,
                box['ymin'] * h,
                box['xmax'] * w,
                box['ymax'] * h,
            ] for box in boxes]
            box_np = np.asarray(box_np)
            assert (box_np[:, 0::2] <= w).all() and (box_np[:, 0::2] >= 0).all()
            assert (box_np[:, 1::2] <= h).all() and (box_np[:, 1::2] >= 0).all()

            instances, features = doit(im, box_np)
        else:
            instances, features = doit_without_boxes(im)
            pred_boxes = instances.pred_boxes.tensor.cpu()
            pred_boxes /= torch.Tensor([w, h, w, h])
            boxes = [
                {
                    'xmin': box[0],
                    'ymin': box[1],
                    'xmax': box[2],
                    'ymax': box[3],
                }
                for box in pred_boxes.tolist()]
        features = torch.cat([
            features.cpu(),
            get_box_feature(boxes, im)
        ], dim=1)
        
        vg_class = instances.pred_classes.tolist()
        vg_class_name = [vg_classes[i] for i in instances.pred_classes.tolist()]
        vg_attr_class = instances.attr_classes.tolist()
        vg_attr_class_name = [vg_attrs[i] for i in instances.attr_classes.tolist()]
        vg_attr_score = instances.attr_scores.tolist()
        
        new_boxes = [{
            **box,
            'vg_class': vg_class[b],
            'vg_class_name': vg_class_name[b],
            'vg_attr_class': vg_attr_class[b],
            'vg_attr_class_name': vg_attr_class_name[b],
            'vg_attr_score': vg_attr_score[b],
        } for b, box in enumerate(boxes)]
        new_anno = {
            **img_anno,
            'boxes_and_score': new_boxes,
            'image_shape': [h, w, c],
        }
        new_annos.append(new_anno)
        name2feature[img_name] = {
            'features': features,
            'anno': new_anno,
            'seed': random_seed,
        }
        
        logger.info(f"{i}/{len(box_anno)}, {im.shape}")
    
    pt_path = output_path
    torch.save(name2feature, pt_path)

    if not augment:
        js_path = os.path.join(
            os.path.dirname(output_path),
            'hateful_boxes_attr.json')
        with open(js_path, mode='w') as jf:
            json.dump(new_annos, jf)


if __name__ == "__main__":
    with logger.catch(reraise=True):
        # oid_boxes(
        #     '/home/ron/Downloads/hateful_meme_data_phase2/box_annos.json',
        #     '/home/ron/Downloads/hateful_meme_data_phase2',
        #     f'/home/ron/Downloads/hateful_meme_data/hateful_memes_v2.pt',
        #     augment=False
        # )
        # for i in range(3):
        #     oid_boxes(
        #         '/home/ron/Downloads/hateful_meme_data_phase2/box_annos.json',
        #         '/home/ron/Downloads/hateful_meme_data_phase2',
        #         f'/home/ron/Downloads/hateful_meme_data/hateful_memes_v2_1122.aug.{i}.pt',
        #         augment=True
        #     )
        fire.Fire({
            'extract_oid_boxes_feat': oid_boxes,
            'extract_oid_boxes_feat_with_img_aug': functools.partial(oid_boxes, augment=True),
        })
