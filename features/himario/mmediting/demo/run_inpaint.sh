
IMG_IN="/home/ron/Downloads/hateful_meme_data_phase2/img_mask_3px/$1.png"
IMG_MASK="/home/ron/Downloads/hateful_meme_data_phase2/img_mask_3px/$1.mask.png"
echo $IMG_IN
echo $IMG_MAKS

python3 demo/inpainting_demo.py \
    configs/inpainting/global_local/gl_256x256_8x12_celeba.py \
    ckpt/gl_256x256_8x12_places_20200619-52a040a8.pth \
    $IMG_IN $IMG_MASK test_gl.png

python3 demo/inpainting_demo.py \
    configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py  \
    ckpt/deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
    $IMG_IN $IMG_MASK test_deepfillv2.png