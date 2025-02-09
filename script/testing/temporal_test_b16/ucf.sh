ROOT=/home/yuyating/workspace/STDD
CKPT=/data3/yuyating/CKPT

#OUT_DIR=$CKPT/basetraining/froster/vitb16_8x16/testing/ucf
#LOAD_CKPT_FILE=$CKPT/basetraining/froster/wa_checkpoints/swa_2_22.pth

# OpenVCLIP
OUT_DIR=$CKPT/basetraining/vitb16_8x16/testing/ucf
LOAD_CKPT_FILE=/data3/yuyating/CKPT/openvclip-b16/swa_2_22.pth

# reproduce OpenVCLIP
#OUT_DIR=$CKPT/basetraining/vitb16_8x16/MAML/testing/ucf
#LOAD_CKPT_FILE=/data3/yuyating/CKPT/basetraining/vitb16_8x16/MAML/checkpoints/checkpoint_epoch_00022.pyth

# channel=12 + IWR
#OUT_DIR=$CKPT/SSTD/testing/ucf
#LOAD_CKPT_FILE=/data3/yuyating/CKPT/SSTD/vitb16_8x16/checkpoints/checkpoint_epoch_00022.pyth

PATCHING_RATIO=0.2

cd $ROOT
export PYTHONPATH=$ROOT/slowfast:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3 \
    TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/ucf101_split1 \
    DATA.PATH_PREFIX /data3/yuyating/dataset/OpenDataLab___UCF101/raw/videos_se256 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/ucf101-index2cls.json \
    DATA.TEXT_AUG False \
    DATA.TEXT_AUG_FEATURE_FILE $CKPT/text_feats/classes_feats_ucf101_tpl8_xmix_ViT-B_16.pt \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TRAIN.BATCH_SIZE 24 \
    TEST.BATCH_SIZE 4 \
    NUM_GPUS 2 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 101 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 1 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    MODEL.KEEP_RAW_MODEL False \
    MODEL.ENSEMBLE_PRED False \
    MODEL.ENSEMBLE_RAWMODEL_RATIO 1.0 \
    TEST.CLIP_ORI_PATH /home/yuyating/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 2 \
    MODEL.TEMPORAL_MODELING_TYPE 'stcross_attend' \
    MODEL.CHANNEL_FOLD 32 \
    MODEL.TEMPORAL_SCALE [1,2] \




