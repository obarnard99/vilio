DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"

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
    python FairFace/inference.py map_race_to_person_box \
        $FEATURE_DIR/img_clean \
        $FEATURE_DIR/box_annos.json \
        $FEATURE_DIR/face_race_boxes.json
else
    echo "[FairFace] found box_annos.race.json"
fi;
