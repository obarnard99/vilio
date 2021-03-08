import os
import glob
import json
import mmcv
import fire
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector

# config_file = 'configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py'
# checkpoint_file = '/Disk2/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth'

# img_dir = '/Disk2/hateful_memes/img_clean'
# out_dir = '/Disk2/hateful_memes/split_img_clean'
# anno_out = '/Disk2/hateful_memes/split_img_clean_boxes.json'


def main(img_dir, out_dir, anno_out, 
        config_file='configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py',
        checkpoint_file='/Disk2/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth'):
    imgs = glob.glob(os.path.join(img_dir, '*.png'))
    os.makedirs(out_dir, exist_ok=True)

    print(F"Find {len(imgs)} imgs!")
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    split_img_counter = 0

    crop_anno = {}
    for imid, img in enumerate(imgs):
        # test a single image and show the results
        # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
        result = inference_detector(model, img)
        det_img_clips = result[0]
        crops = []
        crops_anno = []
        for box in det_img_clips:
            if box[-1] > 0.5:
                crops.append(box)
                crops_anno.append(box.tolist())
            else:
                break
        img_name = os.path.basename(img)
        crop_anno[img_name] = crops_anno
        if len(crops) > 1:
            split_img_counter += 1
            print('-' * 100)
            print(f"{split_img_counter}/{imid}/{len(imgs)}", img)
            pil_img = Image.open(img)
            w, h = pil_img.size
            if w > h:
                crops = sorted(crops, key=lambda x: x[0])
                for j in range(len(crops) - 1):
                    crops[j][2] = (crops[j][2] + crops[j + 1][0]) / 2
            else:
                crops = sorted(crops, key=lambda x: x[1])
                for j in range(len(crops) - 1):
                    crops[j][3] = (crops[j][3] + crops[j + 1][1]) / 2
            
            for j, crop in enumerate(crops):
                print(crop)
                box = crop.astype(np.int32)
                img_name = os.path.basename(img)
                out_img_path = os.path.join(out_dir, img_name.replace('.png', f'.{j}.png'))
                pil_img.crop(
                    [max(box[0], 0),
                    max(box[1], 0),
                    min(box[2], w),
                    min(box[3], h),]
                ).save(out_img_path)
            # import pdb; pdb.set_trace()
        elif len(crops) == 1:
            pass
            

    # visualize the results in a new window
    # model.show_result(img, result)
    with open(anno_out, 'w') as f:
        json.dump(crop_anno, f)


if __name__ == "__main__":
    fire.Fire(main)