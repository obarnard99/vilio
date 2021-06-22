import os
import glob
import json
import fire
import random
from multiprocessing import Pool
from collections import Counter

import dlib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms
from loguru import logger
from bbox import draw_boxes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dlib_model():
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = torch.nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
        torch.load('face_recognition_fair_face_models/fair_face_models/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu'))
    )
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    # model_fair_4 = torchvision.models.resnet34(pretrained=True)
    # model_fair_4.fc = torch.nn.Linear(model_fair_4.fc.in_features, 18)
    # model_fair_4.load_state_dict(
    #     torch.load('fair_face_models/fairface_alldata_4race_20191111.pt', map_location=torch.device('cpu'))
    # )
    # model_fair_4 = model_fair_4.to(device)
    # model_fair_4.eval()
    return cnn_face_detector, sp, model_fair_7


def dlib_detect(img, cnn_face_detector, sp, 
                default_max_size=800, size=300, padding=0.25):
    # plt.imshow(img)
    # plt.show()

    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
#     img = dlib.resize_image(img, rows=new_height, cols=new_width)

    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)

    print(f"Find {num_faces} faces")
#         continue
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    face_boxes = []
    for detection in dets:
        rect = detection.rect
        print(rect.height(), rect.tl_corner(), rect.br_corner())
        face_boxes.append([
            rect.tl_corner().x,
            rect.tl_corner().y,
            rect.br_corner().x,
            rect.br_corner().y,
        ])
        faces.append(sp(img, rect))
    if len(dets) > 0:
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
    else:
        images = []
    
    # for face_img in images:
    #     plt.imshow(face_img)
    #     plt.show()
    return images, face_boxes


def predidct_age_gender_race(images, model_fair_7):
    race_7 = [
        'White',
        'Black',
        'Latino_Hispanic',
        'East Asian',
        'Southeast Asian',
        'Indian',
        'Middle Eastern',
    ]
    gender = [
        'Male',
        'Female'
    ]
    
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    predicts = []
    for index, image in enumerate(images):
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = race_7[np.argmax(race_score)]
        gender_pred = gender[np.argmax(gender_score)]
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)
        
        print(race_pred, gender_pred)
        print(race_score, gender_score)
        predicts.append((race_pred, gender_pred))
    return predicts


def box_coverage(box_a, box_b):
    assert (box_a[2] > 1 and box_b[2] > 1) or (box_a[2] <= 1 and box_b[2] <= 1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    larger = box_a if area_a > area_b else box_b
    small = box_b if area_a > area_b else box_a

    w = small[2] - small[0]
    h = small[3] - small[1]
    ocr_l_to_img_r = max(min(larger[2] - small[0], w), 0)
    ocr_r_to_img_l = max(min(small[2] - larger[0], w), 0)
    cover_w = min(ocr_l_to_img_r, ocr_r_to_img_l)
    
    ocr_t_to_img_b = max(min(larger[3] - small[1], h), 0)
    ocr_b_to_img_t = max(min(small[3] - larger[1], h), 0)
    cover_h = min(ocr_t_to_img_b, ocr_b_to_img_t)
    return (cover_h * cover_w) / (w * h)


def converage_nms(primary_set, sec_set, indies=False, drop=True, threshold=0.4):
    keep_sec = []
    keep = []
    for j, s_box in enumerate(sec_set):
        covers = [0]
        for i, p_box in enumerate(primary_set):
            cov = box_coverage(p_box, s_box)
            covers.append(cov)
        
        if drop:
            if max(covers) < threshold:
                keep_sec.append(s_box)
                keep.append(j)
        else:
            if max(covers) >= threshold:
                keep_sec.append(s_box)
                keep.append(j)
    if indies:
        return keep_sec, keep
    else:
        return keep_sec


def refine_hair_box(box, imh, imw):
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = (box[2] + box[0]) / 2
    cy = (box[3] + box[1]) / 2
    
    new_box = []
    new_side = h

    if h / w > 1.2:
        # long haired
        new_side = h
    else:
        new_side = w
        cy += w // 2
    new_box = [
        max(0, cx - new_side / 2),
        max(0, cy - new_side / 2),
        min(imw, cx + new_side / 2),
        min(imh, cy + new_side / 2),
    ]
    return new_box



def predict(anno, meme_img_dir, cnn_face_detector, sp, model_fair_7, debug=False):
    img_name = anno['img_name']
    img_path = os.path.join(meme_img_dir, img_name)
    img = dlib.load_rgb_image(img_path)
    h, w = img.shape[:2]
    
    face_crops, dlib_boxes = dlib_detect(img, cnn_face_detector, sp)
    
    for bs in anno['boxes_and_score']:
        if 'box' not in bs:
            # mul_w = w if bs['xmax'] < 1 else 1.0
            # mul_h = h if bs['ymax'] < 1 else 1.0
            mul_w = w
            mul_h = h
            bs['box'] = [
                bs['xmin'] * mul_w,
                bs['ymin'] * mul_h,
                bs['xmax'] * mul_w,
                bs['ymax'] * mul_h
            ]
    
    face_box = [
        [int(b) for b in bs['box'][:4]]
        for bs in anno['boxes_and_score']
        if bs['class_name'].lower() in ['face', 'human face']
    ]
    face_box = converage_nms(dlib_boxes, face_box)
    
    head_box = [
        [int(b) for b in bs['box'][:4]]
        for bs in anno['boxes_and_score']
        if bs['class_name'] in ['head', 'human head']
    ]
    head_box = converage_nms(dlib_boxes + face_box, head_box)
    
    hair_box = [
        [int(b) for b in refine_hair_box(bs['box'][:4], h, w)]
        for bs in anno['boxes_and_score']
        if bs['class_name'] in ['hair', 'human hair']
    ]
    hair_box = converage_nms(dlib_boxes + face_box + head_box, hair_box)
    gqa_det_crops = [
        img[b[1]: b[3], b[0]: b[2], ...]
        for b in face_box + head_box
    ]

    if debug:
        _boxes = np.array(dlib_boxes + face_box + head_box)
        _img = img.copy()
        draw_boxes(
            _img,
            _boxes,
            [f"{i}".encode() for i in range(len(_boxes))],
            [1.0] * len(_boxes),
        )
        plt.figure(figsize=(15, 15))
        plt.imshow(_img)
        plt.title(f"{i}")
        plt.show()

    all_crop_boxes = dlib_boxes + face_box + head_box
    all_cls_pred = predidct_age_gender_race(face_crops + gqa_det_crops, model_fair_7)
    assert len(all_crop_boxes) == len(all_cls_pred)
    return all_crop_boxes, all_cls_pred


def detect_gqa_race(gqa_box_anno, meme_img_dir, debug=False):
    with open(gqa_box_anno, 'r') as f:
        gqa_boxes = json.load(f)
    
    target_cls = ['face', 'haed', 'hair']
    cnn_face_detector, sp, model_fair_7 = load_dlib_model()

    for i, anno in enumerate(gqa_boxes):
        if i < 27 and debug:
            continue
        crop_boxes, cls_pred = predict(anno, meme_img_dir, cnn_face_detector, sp, model_fair_7)


def _detect_race(args):
    gqa_boxes, meme_img_dir = args
    cnn_face_detector, sp, model_fair_7 = load_dlib_model()
    results = {}
    with logger.catch(reraise=True):
        for i, anno in enumerate(gqa_boxes):
            crop_boxes, cls_pred = predict(anno, meme_img_dir, cnn_face_detector, sp, model_fair_7)
            results[anno['img_name']] = {
                'face_boxes': crop_boxes,
                'face_race': [c[0] for c in cls_pred],
                'face_gender': [c[1] for c in cls_pred],
            }
    return results


def detect_gqa_race_mp(gqa_box_anno, meme_img_dir, output_path, debug=False, worker=4):
    torch.multiprocessing.set_start_method('spawn')
    with open(gqa_box_anno, 'r') as f:
        gqa_boxes = json.load(f)
        # gqa_boxes = gqa_boxes[:36]
        random.shuffle(gqa_boxes)
    
    arg_split = [(gqa_boxes[i::worker*2], meme_img_dir) for i in range(worker*2)]
    gather_result = {}
    
    with Pool(worker) as pool:
        result_list = pool.map(_detect_race, arg_split)
        for r in result_list:
            gather_result.update(r)
    
    with open(output_path, 'w') as f:
        json.dump(gather_result, f)
    

def map_race_to_person_box(img_dir, boxes_json, face_race_json, detector='oid'):
    assert detector in ['oid', 'gqa']

    person_cls = [
        'Woman',
        'Person',
        'Human body',
        'Man',
        'Girl',
        'Boy',
    ] if detector == 'oid' else [
        ''
    ]
    person_cls = [c.lower() for c in person_cls]

    with open(boxes_json, 'r') as f:
        det_boxes = json.load(f)
    with open(face_race_json, 'r') as f:
        face_det_boxes = json.load(f)
    
    match_cnt = []
    for img_boxes in det_boxes:
        dets = img_boxes['boxes_and_score']
        
        img_name = img_boxes['img_name']
        img_path = os.path.join(img_dir, img_name)
        img = dlib.load_rgb_image(img_path)
        h, w = img.shape[:2]
        
        face_dets = face_det_boxes[img_name]
        face_box = face_dets['face_boxes']
        face_race = face_dets['face_race']
        face_gender = face_dets['face_gender']

        zip_box_size = lambda tup: (tup[0][2] - tup[0][0]) * (tup[0][3] - tup[0][1])
        sorted_by_area = sorted(
            zip(face_box, face_race, face_gender),
            key=zip_box_size,
            reverse=True)
        face_box = [tup[0] for tup in sorted_by_area]
        face_race = [tup[1] for tup in sorted_by_area]
        face_gender = [tup[2] for tup in sorted_by_area]
        
        pbox_idx = []
        for i, det in enumerate(dets):
            if det['class_name'].lower() in person_cls:
                detector_box = [
                    det['xmin'] * w, det['ymin'] * h,
                    det['xmax'] * w, det['ymax'] * h
                ]
                _, keep_idx = converage_nms(
                    [detector_box],
                    face_box,
                    indies=True,
                    drop=False,
                    threshold=0.8,
                )

                if keep_idx:
                    det['race'] = face_race[keep_idx[0]]
                    det['gender'] = face_gender[keep_idx[0]]
                    match_cnt.append(len(keep_idx))
                else:
                    det['race'] = None
                    det['gender'] = None
            else:
                det['race'] = None
                det['gender'] = None

    print('Match cnt freq: ', Counter(match_cnt))
    taged_box_anno_path = boxes_json.replace('.json', '.race.json')
    with open(taged_box_anno_path, 'w') as f:
        json.dump(det_boxes, f)



if __name__ == "__main__":
    with logger.catch(reraise=True):
        # detect_gqa_race(
        #     "/home/ron/Downloads/hateful_meme_data_phase2/box_annos.json",
        #     # "/home/ron/Downloads/hateful_meme_data_phase2/box_annos_bottom_up.json",
        #     '/home/ron/Downloads/hateful_meme_data_phase2/img_clean',
        #     debug=True
        # )
        # detect_gqa_race_mp(
        #     '/home/ron/Downloads/hateful_meme_data_phase2/box_annos.json',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/img_clean',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/face_race_boxes.1118.json',
        #     debug=False
        # )
        # detect_gqa_race_mp(
        #     '/home/ron/Downloads/hateful_meme_data_phase2/box_annos_bottom_up.json',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/img_clean',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/face_race_boxes.bottomup.json',
        #     debug=False
        # )
        # detect_gqa_race_mp(
        #     '/data/box_annos.json',
        #     '/data/img_clean',
        #     '/face_race_boxes.1118.json',
        #     debug=False
        # )

        # map_race_to_person_box(
        #     '/home/ron/Downloads/hateful_meme_data_phase2/img',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/race_box_anno_btmup/box_annos.json',
        #     '/home/ron/Downloads/hateful_meme_data_phase2/face_race_boxes.bottomup.json',
        # )
        # map_race_to_person_box(
        #     '/data/img',
        #     '/data/race_box_anno_btmup/box_annos.json',
        #     '/face_race_boxes.bottomup.json',
        # )

        fire.Fire({
            "detect_race_mp": detect_gqa_race_mp,
            "map_race_to_person_box": map_race_to_person_box,
        })