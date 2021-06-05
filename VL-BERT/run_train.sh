DATA_DIR="/home/miproj/4thyr.oct2020/ojrb2/vilio/data"
FEATURE_DIR="$DATA_DIR/features"
MODEL_DIR="$DATA_DIR/models"
CKPT="$MODEL_DIR/checkpoints"
VLBERT_CKPT="$CKPT/vl-bert"

mkdir -p "$VLBERT_CKPT"
bash scripts/init.sh

MODEL_NAME_LIST=("vl-bert-base-v4" "vl-bert-large-v4" "vl-bert-large-v5-race")
CONFIG_NAME_LIST=("base_4x14G_fp32_k8s_v4" "large_4x14G_fp32_k8s_v4" "large_4x14G_fp32_k8s_v5_race")
CKPT_NAME_LIST=("vl-bert_base_res101_cls-0006.model" "vl-bert_large_res101_cls-0009.model" "vl-bert_large_res101_cls-0009.model")

for index in ${!MODEL_NAME_LIST[*]}; do 
    MODEL_NAME="${MODEL_NAME_LIST[$index]}"
    CONFIG_NAME="${CONFIG_NAME_LIST[$index]}"
    CKPT_NAME="${CKPT_NAME_LIST[$index]}"
    echo "[$MODEL_NAME] & [$CONFIG_NAME] --> [$CKPT_NAME]"

    if [ ! -d "$VLBERT_CKPT/$MODEL_NAME" ]; then
        mkdir -p "$VLBERT_CKPT/$MODEL_NAME"
        echo "************  [TRAIN] $MODEL_NAME  ************"
        sh scripts/dist_run_single.sh \
            4 \
            cls/train_end2end.py \
            "cfgs/cls/$CONFIG_NAME.yaml" \
            "checkpoints/$MODEL_NAME"
    else
        echo "Dir \"$VLBERT_CKPT/$MODEL_NAME\" is already exist, assume training is completed..."
    fi;

    if [ -e "$VLBERT_CKPT/$MODEL_NAME/$CONFIG_NAME/train+val_train/$CKPT_NAME" ] && [ ! -e "$VLBERT_CKPT/$MODEL_NAME/${CONFIG_NAME}_cls_test.csv" ]; then
        echo "************  [TEST] $MODEL_NAME  ************"
        python cls/test.py \
            --cfg "cfgs/cls/$CONFIG_NAME.yaml" \
            --ckpt "checkpoints/$MODEL_NAME/$CONFIG_NAME/train+val_train/$CKPT_NAME" \
            --result-path "checkpoints/$MODEL_NAME"
    else
        echo "Checkpoint not found: $VLBERT_CKPT/$MODEL_NAME/$CONFIG_NAME/train+val_train/$CKPT_NAME"
    fi;
    
    if [ -e "$VLBERT_CKPT/$MODEL_NAME/${CONFIG_NAME}_cls_test.csv" ]; then
        echo "************  [CLEAN] $VLBERT_CKPT/$MODEL_NAME/$CKPT_NAME  ************"
        rm -rf checkpoints/$MODEL_NAME/$CONFIG_NAME
    fi;
done;