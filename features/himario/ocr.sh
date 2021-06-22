DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"

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
