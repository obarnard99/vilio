import glob
import argparse

import os
import mmcv
import torch

from mmedit.apis import init_model, inpainting_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Inpainting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('masked_img_path', help='path to input image file')
    parser.add_argument('mask_path', help='path to input mask file')
    parser.add_argument('save_path', help='path to save inpainting result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    result = inpainting_inference(model, args.masked_img_path, args.mask_path)
    result = tensor2img(result, min_max=(-1, 1))[..., ::-1]

    mmcv.imwrite(result, args.save_path)
    if args.imshow:
        mmcv.imshow(result, 'predicted inpainting result')

def parse_dir_args():
    parser = argparse.ArgumentParser(description='Inpainting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_dir', help='path to input image file')
    parser.add_argument('save_dir', help='path to save inpainting result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main_dir():
    args = parse_dir_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_and_mask = glob.glob(os.path.join(args.img_dir, '*.png'))
    img_map = {}
    mask_map = {}
    for path in img_and_mask:
        name = os.path.basename(path)
        imgid = name.split('.')[0]
        if '.mask' in name:
            mask_map[imgid] = path
        else:
            img_map[imgid] = path

    for i, imgid in enumerate(img_map.keys()):
        print(f"[{i}/{len(img_map)}] {imgid}")
        masked_img_path = img_map[imgid]
        mask_path = mask_map[imgid]
        img_name = os.path.basename(masked_img_path)
        
        result = inpainting_inference(model, masked_img_path, mask_path)
        result = tensor2img(result, min_max=(-1, 1))[..., ::-1]
        h, w = mmcv.imread(masked_img_path).shape[:2]
        result = result[:h, :w, ...]

        save_path = os.path.join(args.save_dir, img_name)
        mmcv.imwrite(result, save_path)
        if args.imshow:
            mmcv.imshow(result, 'predicted inpainting result')


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # main()
    main_dir()
