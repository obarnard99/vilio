DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"

python gcp/web_entity.py create_img_list \
  $FEATURE_DIR/img_clean \
  $FEATURE_DIR/img_list_all_clean \
  --split_size 20000 \
  --exclude_dir $FEATURE_DIR/split_img_clean

python gcp/web_entity.py create_img_list \
  $FEATURE_DIR/split_img_clean \
  $FEATURE_DIR/split_img_list_clean \
  --split_size 20000

mkdir -p "$FEATURE_DIR/entity_json"
bash gcp/loop.sh \
  $FEATURE_DIR/img_list_all_clean/img_clean_split.0.txt \
  $FEATURE_DIR/entity_json

mkdir -p "$FEATURE_DIR/entity_json_split"
bash gcp/loop.sh \
  $FEATURE_DIR/split_img_list_clean/split_img_clean_split.0.txt \
  $FEATURE_DIR/entity_json_split

cp $FEATURE_DIR/entity_json_split/*.json "$FEATURE_DIR/entity_json"

