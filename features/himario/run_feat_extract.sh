DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
UNITER_DIR="$FEATURE_DIR/uniter"

#if [ ! -d $UNITER_DIR ]; then
#    mkdir -p $UNITER_DIR
#fi

#if [ ! -d "$UNITER_DIR/MEME_NPZ" ]; then
#    mkdir -p "$UNITER_DIR/MEME_NPZ"
#    bash -c "python tools/generate_npz.py --gpu 0"
#fi

# if [! -d "$UNITER_DIR/MEME_NPZ_DB"]; then
#     IMG_NPY="$UNITER_DIR/MEME_NPZ"
#     NAME=$(basename $IMG_NPY)
#     docker run --ipc=host --rm -it \
#         --mount src=$(pwd),dst=/src,type=bind \
#         --mount src="$UNITER_DIR/MEME_NPZ_DB",dst=/img_db,type=bind \
#         --mount src=$IMG_NPY,dst=/$NAME,type=bind,readonly \
#         -w /src chenrocks/uniter \
#         python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db
# fi

# Extract bottom up attention roi feature PT file
echo "[py-bottom-up-attention] extract features"
if [ ! -e "$FEATURE_DIR/hateful_memes_v2.pt" ]; then
    python hateful_meme_feature.py \
      $FEATURE_DIR/extract_oid_boxes_feat \
      $FEATURE_DIR//box_annos.json \
      $FEATURE_DIR \
      $FEATURE_DIR/hateful_memes_v2.pt
fi
echo "[py-bottom-up-attention] extract features with augmentation 1/3"
if [ ! -e "$FEATURE_DIR/hateful_memes_v2.aug.0.pt" ]; then
    python hateful_meme_feature.py \
      extract_oid_boxes_feat_with_img_aug \
      $FEATURE_DIR/box_annos.json \
      $FEATURE_DIR \
      $FEATURE_DIR/hateful_memes_v2.aug.0.pt \
      --random_seed 5659
fi
echo "[py-bottom-up-attention] extract features with augmentation 2/3"
if [ ! -e "$FEATURE_DIR/hateful_memes_v2.aug.1.pt" ]; then
    python hateful_meme_feature.py \
      extract_oid_boxes_feat_with_img_aug \
      $FEATURE_DIR/box_annos.json \
      $FEATURE_DIR \
      $FEATURE_DIR/hateful_memes_v2.aug.1.pt \
      --random_seed 6582
fi
echo "[py-bottom-up-attention] extract features with augmentation 3/3"
if [ ! -e "$FEATURE_DIR/hateful_memes_v2.aug.2.pt" ]; then
    python hateful_meme_feature.py \
      extract_oid_boxes_feat_with_img_aug \
      $FEATURE_DIR/box_annos.json \
      $FEATURE_DIR \
      $FEATURE_DIR/hateful_memes_v2.aug.2.pt \
      --random_seed 7505
fi