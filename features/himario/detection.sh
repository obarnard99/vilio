DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"

# Run InceptionV2 OID
if [ ! -e "$FEATURE_DIR/box_annos.json" ]; then
    echo "[TF OID] OpenImageV4 object detector"
    python gen_bbox.py \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/box_annos.json \
else
    echo "[TF OID] found box_annos.json"
fi;
