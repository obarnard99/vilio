DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"

# OCR to get text bbox and mask
if [ ! -e "$FEATURE_DIR/ocr.json" ]; then
    echo "[OCR] detect"
    python ocr.py detect $DATA_DIR
else
    echo "[OCR] found ocr.json"
fi;

if [ ! -e "$FEATURE_DIR/ocr.box.json" ]; then
    echo "[OCR] convert point annotation to box"
    python ocr.py point_to_box $FEATURE_DIR/ocr.json
else
    echo "[OCR] found ocr.box.json"
fi;


if [ ! -d "$FEATURE_DIR/img_mask_3px" ]; then
    echo "[OCR] create text segmentation mask"
    python ocr.py generate_mask \
        $FEATURE_DIR/ocr.box.json \
        $DATA_DIR/img \
        $FEATURE_DIR/img_mask_3px
else
    echo "[OCR] found img_mask_3px"
fi;

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

# Run InceptionV2 OID
if [ ! -e "$FEATURE_DIR/box_annos.json" ]; then
    echo "[TF OID] OpenImageV4 object detector"
    python gen_bbox.py \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/box_annos.json
else
    echo "[TF OID] found box_annos.json"
fi;

# Detect and extract image patchs
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

# Get race of face and head
if [ ! -e "$FEATURE_DIR/face_race_boxes.json" ]; then
    echo "[FairFace] detect ethnicity"
    python FairFace/inference.py detect_race_mp \
        $FEATURE_DIR/box_annos.json \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/face_race_boxes.json \
        --debug False \
        --worker 4
else
    echo "[FairFace] found face_race_boxes.json"
fi;

if [ ! -e "$FEATURE_DIR/box_annos.race.json" ]; then
    echo "[FairFace] map ethnicity to person box"
    python inference.py map_race_to_person_box \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/box_annos.json \
        $FEATURE_DIR/face_race_boxes.json
else
    echo "[FairFace] found box_annos.race.json"
fi;
