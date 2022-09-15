import os
import tempfile
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union
from torch.cuda.amp import autocast, GradScaler

import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchinfo import summary as TISummary
import numpy as np
from datetime import datetime
from tqdm import trange
from tqdm import tqdm, tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from logging import handlers

# from PyUtils.PLModuleInterface import PLMInterface, GraphCallback
from PyUtils.logs.print import *

from src.config.config_v1 import CONFIGS
from src.dataset.dataset import YoloDataset
from src.utils.voc_utils import get_anchors
from src.nets.yolo import YoloBody
import src.loss.yolov5_loss as YoloLoss

import os
import argparse
import torch
from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.utils.tensorboard.writer as TBWriter
from prettytable import PrettyTable
from PyUtils.utils.meter import AverageMeter
from PyUtils.pytorch.callback import LogCallback, TrainCallback, GraphCallback
from PyUtils.pytorch.module import TrainModule, Trainer



class YoloV5TrainModule(TrainModule):

    def __init__(self, pretrained=None):
        super(YoloV5TrainModule, self).__init__(pretrained)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None
        self.val_dataset = None
        self.val_sampler = None
        self.val_loader = None
        self.test_dataset = None
        self.test_sampler = None
        self.test_loader = None
        self.pretrained = pretrained

    def create_model(self, local_rank):
        # create local model
        phi = 's'
        backbone = 'cspdarknet'
        pretrained = False
        self.model = YoloBody(
            CONFIGS.ANCHORS_MASK,
            CONFIGS.VOC_CLS_NUM,
            phi,
            backbone,
            pretrained=pretrained,
            input_shape=(CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE)
        ).to(local_rank)

        pretrained = f'/data/ylw/code/pl_yolo_v5/checkpoints/1012.pth'
        if pretrained and os.path.exists(pretrained) and local_rank == 0:
            sllog << f'[{local_rank}]  loading pre-trained model[{pretrained}] ...'
            # self.model.load_state_dict(torch.load(pretrained).module.state_dict())
            self.model.load_state_dict(torch.load(pretrained))
            sllog << f'[{local_rank}]  load pre-trained model complete.'
            
        dist.barrier()

    def create_loss(self):
        self.criterion = YoloLoss.YOLOLoss(
            np.array(CONFIGS.ANCHORS),
            CONFIGS.VOC_CLS_NUM,
            (
                CONFIGS.TRAIN_IMG_SIZE,
                CONFIGS.TRAIN_IMG_SIZE
            ),
            CONFIGS.ANCHORS_MASK,
            label_smoothing=0
        )

    def create_optim(self, model):
        # self.optimizer = optim.SGD(
        #     model.parameters(), 
        #     lr=CONFIGS.TRAIN_LR_INIT,
        #     momentum=CONFIGS.TRAIN_MOMENTUM, 
        #     weight_decay=CONFIGS.TRAIN_WEIGHT_DECAY
        # )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=CONFIGS.TRAIN_LR_INIT,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=CONFIGS.TRAIN_WEIGHT_DECAY,
            amsgrad=False
        )
        
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 
        #     mode='min', 
        #     factor=0.1, 
        #     patience=10,
        #     threshold=1e-4, 
        #     threshold_mode='rel', 
        #     cooldown=0,
        #     min_lr=0, 
        #     eps=1e-8, 
        #     verbose=False
        # )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=CONFIGS.TRAIN_LR_END)

    def create_data_loader(self):
        train_annotation_path = os.path.join(CONFIGS.DATASET_DIR, '2007_train.txt')
        lines, train_lines, val_lines = None, None, None
        with open(train_annotation_path, encoding='utf-8') as f:
            lines = f.readlines()
            train_lines = lines[:int(len(lines)*(1-0.2))]
            val_lines = lines[int(len(lines)*(1-0.2)):]
        train_annotation_path = os.path.join(CONFIGS.DATASET_DIR, '2012_train.txt')
        with open(train_annotation_path, encoding='utf-8') as f:
            new_lines = f.readlines()
            train_lines.extend(new_lines)
            
        self.train_dataset = YoloDataset(
            train_lines, 
            (CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE), 
            CONFIGS.VOC_CLS_NUM, 
            np.array(CONFIGS.ANCHORS), 
            CONFIGS.ANCHORS_MASK, 
            mosaic=CONFIGS.DATASET_MOSAIC, 
            mixup=CONFIGS.DATASET_MIXUP, 
            mosaic_prob=CONFIGS.DATASET_MOSAIC_PROB, 
            mixup_prob=CONFIGS.DATASET_MIXUP_PROB, 
            train=True,
            fixed_input_shape=True
        )
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, 
            shuffle = False, 
            batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
            num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
            pin_memory=True,
            drop_last=True, 
            collate_fn=self.train_dataset.yolo_dataset_collate,
            sampler=self.train_sampler
        )
        self.val_dataset = YoloDataset(
            val_lines, 
            (CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE), 
            CONFIGS.VOC_CLS_NUM, 
            np.array(CONFIGS.ANCHORS), 
            CONFIGS.ANCHORS_MASK, 
            mosaic=CONFIGS.DATASET_MOSAIC, 
            mixup=CONFIGS.DATASET_MIXUP, 
            mosaic_prob=CONFIGS.DATASET_MOSAIC_PROB, 
            mixup_prob=CONFIGS.DATASET_MIXUP_PROB, 
            train=False,
            fixed_input_shape=True
        )
        self.val_sampler = DistributedSampler(self.val_dataset)
        self.val_loader = DataLoader(
            self.val_dataset, 
            shuffle = False, 
            batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
            num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
            pin_memory=True,
            drop_last=True, 
            collate_fn=self.val_dataset.yolo_dataset_collate,
            sampler=self.val_sampler
        )

        test_annotation_path = os.path.join(CONFIGS.DATASET_DIR, '2007_val.txt')
        test_lines = None
        with open(test_annotation_path, encoding='utf-8') as f:
            test_lines = f.readlines()
            
        self.test_dataset = YoloDataset(
            test_lines, 
            (CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE), 
            CONFIGS.VOC_CLS_NUM, 
            np.array(CONFIGS.ANCHORS), 
            CONFIGS.ANCHORS_MASK, 
            mosaic=CONFIGS.DATASET_MOSAIC, 
            mixup=CONFIGS.DATASET_MIXUP, 
            mosaic_prob=CONFIGS.DATASET_MOSAIC_PROB, 
            mixup_prob=CONFIGS.DATASET_MIXUP_PROB, 
            train=False,
            fixed_input_shape=True
        )
        self.test_sampler = DistributedSampler(self.test_dataset)
        self.test_loader = DataLoader(
            self.test_dataset, 
            shuffle = False, 
            batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
            num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
            pin_memory=True,
            drop_last=True, 
            collate_fn=self.test_dataset.yolo_dataset_collate,
            sampler=self.test_sampler
        )

    def train_step(self, batch_idx, batch, local_rank):
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self.model(images.to(local_rank))
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(
                l,
                output,
                targets,
                y_trues[l], 
                batch
            )
            loss_all  += loss_item
            losses[l] += loss_item
        return {'loss': loss_all, 'loss1':losses[0], 'loss2':losses[1], 'loss3':losses[2]}

    def train_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def eval_step(self, batch_idx, batch, local_rank):
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self.model(images.to(local_rank))
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(
                l,
                output,
                targets,
                y_trues[l],
                batch
            )
            loss_all  += loss_item
            losses[l] += loss_item
        return {'loss': loss_all, 'loss1':losses[0], 'loss2':losses[1], 'loss3':losses[2]}

    def eval_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def test_step(self, batch_idx, batch, local_rank):
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self.model(images.to(local_rank))
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(
                l,
                output,
                targets,
                y_trues[l],
                batch
            )
            loss_all  += loss_item
            losses[l] += loss_item.item()
        return {'loss': loss_all, 'loss1':losses[0], 'loss2':losses[1], 'loss3':losses[2]}

    def test_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def predict(self, images, device_id):
        preds = self.model(images.to(device_id))
        return preds

    def predict_end(self, predicts, device_id):
        ...

    def set_callbacks(self):
        graph_cb = TrainCallback(
            interval=10, 
            log_dir='/data/ylw/code/pl_yolo_v5/logs/yolov5_000001', 
            dummy_input=np.zeros(shape=(2, 3, 446, 446))
        )
        log_cbs = LogCallback(
            meters={
                'loss': AverageMeter(name='loss', fmt=':4f'),
                'loss1': AverageMeter(name='loss1', fmt=':4f'),
                'loss2': AverageMeter(name='loss2', fmt=':4f'),
                'loss3': AverageMeter(name='loss3', fmt=':4f')
            }, 
            log_dir=f'/data/ylw/code/pl_yolo_v5/runs', log_surfix='default'
        )
        return [graph_cb, log_cbs]



if __name__=="__main__":
    train_module = YoloV5TrainModule()
    
    trainer = Trainer(
        train_module=train_module, 
        configs=CONFIGS,
        mode='train', 
        accelarate=2, 
        precision=True,
        grad_average=False,
        sync_bn=True
    )
    trainer.fit()
