#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.TEST.UPDATE_STATE = False
    _C.TEST.CUSTOM_LOAD = False
    _C.TEST.CUSTOM_LOAD_FILE = None
    _C.TRAIN.CUSTOM_LOAD = False
    _C.TRAIN.CUSTOM_LOAD_FILE = None
    _C.VAL_MODE = False
    _C.TEST.OPENSET = False
    _C.IMAGENET_SIMPLELABEL_PATH = None
    _C.TEST.PATCHING_MODEL = False
    _C.TEST.CLIP_ORI_PATH = None
    _C.TEST.PATCHING_RATIO = 0.5
    _C.TRAIN.ZS_RESTART_CONS = False
    _C.TRAIN.ZS_INIT_CONS = False
    _C.TRAIN.ZS_CONS = False
    _C.TRAIN.ZS_RESTART_EPOCH = -1
    _C.TRAIN.ZS_CONS_RATIO = 0.8
    _C.TRAIN.ADAPT_ZS_CONS_RATIO = False
    _C.TRAIN.CLIP_ORI_PATH = None
    _C.TRAIN.ZERO_SHOT_META_LEARN = False
    _C.DATA.INDEX_LABEL_MAPPING_FILE = ''
    _C.DATA.TEXT_AUG = False
    _C.DATA.TEXT_AUG_FEATURE_FILE = ''
    _C.TUNE_HEAD = False
    _C.MODEL.TEMPORAL_MODELING_TYPE = None
    _C.MODEL.USE_CHECKPOINT = False
    _C.MODEL.STATIC_GRAPH = False
    _C.MODEL.TEXT_PROMPT = False
    _C.MODEL.PROMPT_NUM = 1
    _C.MODEL.CONTEXT_LENGTH = 77
    _C.MODEL.NUM_EXPERTS = 0
    _C.MODEL.EXPERT_INSERT_LAYERS = [10,11]
    _C.MODEL.FINETUNE_FACTOR = 1.0
    _C.MODEL.ADAPT_FINETUNE_FACTOR = 1.0
    _C.MODEL.MLP_FINETUNE_FACTOR = 1.0
    _C.MODEL.EXPERT_FINETUNE_FACTOR = 1.0
    _C.MODEL.DEFAULT_FINETUNE_FACTOR = 1.0
    _C.MODEL.ROUTING_FINETUNE_FACTOR = 1.0
    _C.MODEL.RECORD_ROUTING = False
    _C.MODEL.ROUTING_FREQUENCE_CONSTRAIN = 0.5 
    _C.MODEL.ROUTING_FREQ_CONS_FACTOR = 1.0 
    _C.MODEL.CLS_LOSS_RATIO = 1.0
    _C.MODEL.ROUTING_TYPE = 'patch-level'
    _C.MODEL.LOSS_FREQ_TYPE = "mse"
    _C.MODEL.KEEP_RAW_MODEL = False
    _C.MODEL.ENSEMBLE_PRED = False
    _C.MODEL.ENSEMBLE_RAWMODEL_RATIO = 0.0
    _C.MODEL.MASK_RATE = 0.5
    _C.MODEL.MASK_STRIDE = [1,2,2]
    _C.MODEL.SPATIAL_REPEAT = True
    _C.MODEL.TEMPORAL_SHUFFLE = True
    _C.MODEL.CHANNEL_FOLD = 64
    _C.MODEL.TEMPORAL_SCALE = [1]
    _C.TRAIN.EWC_SET = False
    _C.TRAIN.EWC_CONSTRAIN_RATIO = 1.0
    _C.TRAIN.EWC_LOAD_FILE = None
    _C.TRAIN.EWC_IDENTITY_FISHER = False
    _C.TRAIN.EWC_IGNORE_LOGIT_SCALE = False
    _C.MODEL.RAW_MODEL_DISTILLATION = False
    _C.MODEL.DISTILLATION_RATIO = 1.0 
    _C.SOLVER.COSINE_RESTART_EPOCH = 0
    _C.TRAIN.LINEAR_CONNECT_CLIMB = False
    _C.TRAIN.LINEAR_CONNECT_LOSS_RATIO = 1.0
    _C.TRAIN.LINEAR_CONNECT_SAMPLE = True
    _C.TRAIN.LINEAR_CONNECT_SAMPLE_L = 0.4
    _C.TRAIN.LINEAR_CONNECT_SAMPLE_R = 0.6
    # pass


