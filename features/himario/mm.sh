DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"

# Remove text by inpainting
if [ ! -d "$FEATURE_DIR/img_clean" ]; then
    echo "[mmediting] remove text using text mask"
    python mmediting/demo/inpainting_demo.py \
        mmediting/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py \
        $MODEL_DIR/deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
        $FEATURE_DIR/img_mask_3px \
        $FEATURE_DIR/img_clean
else
    echo "[mmediting] found img_clean"
fi;

# Detect and extract image patches
if [ ! -e "$FEATURE_DIR/split_img_clean_boxes.json" ]; then
    echo "[mmdetection] image patch detection"
    python mmdetection/tools/inspect_image_clip.py \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/split_img_clean \
        $FEATURE_DIR/split_img_clean_boxes.json \
        --config_file mmdetection/configs/res2net/faster_rcnn_r2_101_fpn_2x_img_clip.py \
        --checkpoint_file $MODEL_DIR/faster_rcnn_r2_101_fpn_2x_img_clip/epoch_3.pth
else
    echo "[mmdetection] found split_img_clean_boxes.json"
fi;
