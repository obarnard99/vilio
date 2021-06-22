DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
ANNO_DIR="$DATA_DIR/annotations"
ENTITY_DIR="$FEATURE_DIR/entity_json"

export GOOGLE_APPLICATION_CREDENTIALS="/home/miproj/4thyr.oct2020/ojrb2/uploads/f-mt126-1-e8ab23b3ed9a.json"

if [ ! -e "$FEATURE_DIR/entity_tags.pickle" ]; then
    echo "[web_entity] create image entity description"
    python gcp/web_entity.py \
      create_description \
      --json_dir $ENTITY_DIR \
      --out_pickle $FEATURE_DIR/entity_tags.pickle
fi
if [ ! -e "$FEATURE_DIR/entity_cleaned.pickle" ]; then
    echo "[web_entity] cleaning image entity description"
    python gcp/web_entity.py \
      titles_cleanup \
      $FEATURE_DIR/entity_tags.pickle \
      --out_pickle $FEATURE_DIR/entity_cleaned.pickle
fi

if [ ! -e "$FEATURE_DIR/summary_entity_cleaned.pickle" ]; then
    echo "[web_entity] summary image entity description"
    python gcp/web_entity.py \
      titles_summary \
      $FEATURE_DIR/entity_cleaned.pickle \
      $FEATURE_DIR/summary_entity_cleaned.pickle
fi

echo "Build: $ANNO_DIR/dev_all.jsonl"
    python ../../utils/pandas_scripts.py clean_data \
      --data_path $ANNO_DIR \
      --force

SPLIT_LIST=("train" "test_unseen" "test_seen" "dev_unseen" "dev_seen" "dev_all")

for SPLIT in "${SPLIT_LIST[@]}"; do
    if [ ! -e "$ANNO_DIR/$SPLIT.entity.jsonl" ]; then
        echo "Insert features to: $ANNO_DIR/$SPLIT.jsonl"
        python gcp/web_entity.py insert_anno_jsonl  \
          $FEATURE_DIR/summary_entity_cleaned.pickle  \
          $ANNO_DIR/$SPLIT.jsonl \
          $FEATURE_DIR/split_img_clean_boxes.json \
          $FEATURE_DIR/ocr.box.json \
          --img_dir $DATA_DIR/img
    fi
done

cp "$ANNO_DIR/train.entity.jsonl" "$ANNO_DIR/train_dev_all.entity.jsonl"
cat "$ANNO_DIR/dev_all.entity.jsonl" >> "$ANNO_DIR/train_dev_all.entity.jsonl"