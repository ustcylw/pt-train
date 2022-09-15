import os, sys
import torch


class CONFIGS:
    PRE_SYMBOL = 'yolov5'
    POST_SYMBOL = '000001'

    MODE = 'train'  # 'train'  'val'  'test'
    SEED = 7
    

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f'[************]  PROJECT DIR: {ROOT_DIR}')
    if not os.path.exists(ROOT_DIR):
        raise ValueError(f'PROJECT NOT EXIST!!! {ROOT_DIR}')
    DATASET_DIR = '/data/ylw/code/pl_yolo_v5/data'
    CHECKPOINT_DIR = os.path.join(os.path.dirname(ROOT_DIR), f'checkpoints/{PRE_SYMBOL}{POST_SYMBOL}')
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    LOGS_DIR = os.path.join(ROOT_DIR, f'logs/{PRE_SYMBOL}_{POST_SYMBOL}')
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)


    # AVAIL_GPUS = min(1, torch.cuda.device_count())
    



    DATASET_MOSAIC = True
    DATASET_MOSAIC_PROB = 0.5
    DATASET_MIXUP = True
    DATASET_MIXUP_PROB = 0.5
    DATASET_SPECIAL_AUG_RATIO = 0.7

    # train
    TRAIN_START_EPOCH = 0
    TRAIN_EPOCHS = 80
    TRAIN_IMG_SIZE = 448
    TRAIN_AUGMENT = True
    TRAIN_BATCH_SIZE = 12
    TRAIN_MULTI_SCALE_TRAIN = True
    TRAIN_IOU_THRESHOLD_LOSS = 0.5
    TRAIN_START_EPOCHS = 0
    TRAIN_EPOCHS = 50
    TRAIN_NUMBER_WORKERS = 2
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 0.0005
    TRAIN_LR_INIT = 1e-4
    TRAIN_LR_END = 1e-6
    TRAIN_WARMUP_EPOCHS = 2  # or None
    TRAIN_PRETRAINED_MODEL_NAME = 'last_.ckpt'
    TRAIN_PRETRAINED_DIR = os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}')
    TRAIN_INPUT_SHAPE = [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, 3]
    TRAIN_LOG_INTERVAL = 60
    
    # test
    TEST_IMG_SIZE = 544
    TEST_BATCH_SIZE = 1
    TEST_NUMBER_WORKERS = 0
    TEST_CONF_THRESH = 0.01
    TEST_NMS_THRESH = 0.5
    TEST_SCORE_THRESH = 0.5
    TEST_MULTI_SCALE_TEST = False
    TEST_FLIP_TEST = False
    TEST_SAVE = True
    TEST_SAVE_DIR = os.path.join(ROOT_DIR, 'results')
    TEST_SHOW = True
    TEST_PRETRAINED_MODEL_NAME = os.path.join(os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}'), 'epoch=79-val_loss=16.26-other_metric=0.00.ckpt')  #  'last.ckpt'
