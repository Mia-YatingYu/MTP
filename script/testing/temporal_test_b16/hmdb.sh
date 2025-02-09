ROOT=/home/yuyating/GitHub/Open-VCLIP
CKPT=/data3/yuyating/CKPT

# froster
#OUT_DIR=$CKPT/basetraining/froster/vitb16_8x16/testing/hmdb
#LOAD_CKPT_FILE=$CKPT/basetraining/froster/wa_checkpoints/swa_2_22.pth

#OUT_DIR=$CKPT/basetraining/vitb16_8x16/testing/hmdb
#LOAD_CKPT_FILE=/data3/yuyating/CKPT/openvclip-b16/swa_2_22.pth

# ours
OUT_DIR=$CKPT/expand_FOMAML_virtual/vitb16_8x16/testing/hmdb
LOAD_CKPT_FILE=$CKPT/expand_FOMAML_virtual/vitb16_8x16/wa_checkpoints/gwa_0_22.pth
PATCHING_RATIO=0.5

cd $ROOT
export PYTHONPATH=$ROOT/slowfast:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/hmdb_split2 \
    DATA.PATH_PREFIX /data2/yuyating/dataset/OpenDataLab___HMDB51/raw/data \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/hmdb-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TRAIN.BATCH_SIZE 24 \
    TEST.BATCH_SIZE 256 \
    NUM_GPUS 4 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 51 \
    MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
    MODEL.KEEP_RAW_MODEL False \
    MODEL.ENSEMBLE_PRED False \
    MODEL.ENSEMBLE_RAWMODEL_RATIO 0.5 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL True \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH /home/yuyating/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 2 \


