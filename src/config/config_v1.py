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
    

    COLOR = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]
    ]

    VOC_CLASS = [
        'aeroplane', 'bicycle', 'bird', 
        'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 
        'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]
    VOC_CLS_NUM = 20

    ANCHORS = [[ 10.,  13.],
       [ 16.,  30.],
       [ 33.,  23.],
       [ 30.,  61.],
       [ 62.,  45.],
       [ 59., 119.],
       [116.,  90.],
       [156., 198.],
       [373., 326.]]
    ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    STRIDES = [8, 16, 32]
    ANCHORS_PER_SCLAE = 3

    DATASET_MOSAIC = True
    DATASET_MOSAIC_PROB = 0.5
    DATASET_MIXUP = False
    DATASET_MIXUP_PROB = 0.5
    DATASET_SPECIAL_AUG_RATIO = 0.7

    # train
    TRAIN_IMG_SIZE = 640
    TRAIN_AUGMENT = True
    TRAIN_BATCH_SIZE = 16
    TRAIN_MULTI_SCALE_TRAIN = True
    TRAIN_IOU_THRESHOLD_LOSS = 0.5
    TRAIN_START_EPOCHS = 1000
    TRAIN_EPOCHS = 200
    TRAIN_SAVE_MODEL_INTERVAL = 10
    TRAIN_NUMBER_WORKERS = 4
    TRAIN_MOMENTUM = 0.9
    TRAIN_WEIGHT_DECAY = 1e-5
    TRAIN_LR_INIT = 1e-4 # 1e-5 # 1e-4  1e-3
    TRAIN_LR_END = 1e-8
    TRAIN_WARMUP_EPOCHS = 2  # or None
    TRAIN_PRETRAINED_MODEL_NAME = '195.ckpt'
    TRAIN_PRETRAINED_DIR = os.path.join(ROOT_DIR, f'checkpoints/{PRE_SYMBOL}_{POST_SYMBOL}')
    TRAIN_INPUT_SHAPE = [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, 3]
    TRAIN_LOG_INTERVAL = 60
    TRAIN_ACC_INTERVAL = 3
    
    # test
    TEST_IMG_SIZE = 640
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
