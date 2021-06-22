import json
import random
import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ImageClip(CustomDataset):

    CLASSES = ('image',)
    MAX_ATTR_PER_BOX = 5

    def load_annotations(self, ann_file):
        # ann_list = mmcv.list_from_file(ann_file)
        with open(ann_file, mode='r') as f:
            ann_list = json.load(f)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if len(ann_line['boxes']) == 0:
                continue

            width = ann_line['width']
            height = ann_line['height']

            bboxes = np.array(ann_line['boxes'], dtype=np.float32)
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 1, width - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 1, height - 1)
            labels = ann_line['class_id']
            data_infos.append(
                dict(
                    filename=ann_line['image_name'],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=bboxes,
                        labels=labels)
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']