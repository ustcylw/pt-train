import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union





import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pytorch_lightning.core.datamodule import LightningDataModule
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchinfo import summary as TISummary
import numpy as np

from PyUtils.PLModuleInterface import PLMInterface, GraphCallback
from PyUtils.logs.print import *
import PyUtils.PLModuleInterface as PLMI

from src.config.config_v1 import CONFIGS
from src.dataset.dataset import YoloDataset
from src.utils.voc_utils import get_anchors
from src.nets.yolo import YoloBody
import src.loss.yolov5_loss as YoloLoss



import os
import argparse
import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



def run_worker(local_rank, world_size):
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = "INFO"
    
    # create default process group
    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=world_size
    )

    # create local model
    phi = 's'
    backbone = 'cspdarknet'
    pretrained = False
    model = YoloBody(
        CONFIGS.ANCHORS_MASK,
        CONFIGS.VOC_CLS_NUM,
        phi,
        backbone,
        pretrained=pretrained,
        input_shape=(CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE)
    ).to(local_rank)

    pretrained = f'/data/ylw/code/pl_yolo_v5/checkpoints/3.pth'
    if pretrained and os.path.exists(pretrained):
        sllog << f'loading pre-trained model[{pretrained}] ...'
        model.load_state_dict(torch.load(pretrained))
        sllog << f'load pre-trained model complete.'

    dist.barrier()
    
    # construct DDP model
    ddp_model = DDP(model, device_ids=[local_rank])

    # define loss function and optimizer
    criterion = YoloLoss.YOLOLoss(
        np.array(CONFIGS.ANCHORS),
        CONFIGS.VOC_CLS_NUM,
        (
            CONFIGS.TRAIN_IMG_SIZE,
            CONFIGS.TRAIN_IMG_SIZE
        ),
        CONFIGS.ANCHORS_MASK,
        label_smoothing=0
    )
    optimizer = optim.SGD(
        ddp_model.parameters(), 
        lr=CONFIGS.TRAIN_LR_INIT,
        momentum=CONFIGS.TRAIN_MOMENTUM, 
        weight_decay=CONFIGS.TRAIN_WEIGHT_DECAY
    )
    
    train_annotation_path = os.path.join(CONFIGS.DATASET_DIR, '2007_train.txt')
    lines, train_lines, val_lines = None, None, None
    with open(train_annotation_path, encoding='utf-8') as f:
        lines = f.readlines()
        train_lines = lines[:int(len(lines)*(1-0.2))]
        val_lines = lines[int(len(lines)*(1-0.2)):]
        
    train_dataset = YoloDataset(
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
        fixed_input_shape=False
    )
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        shuffle = False, 
        batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
        num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
        pin_memory=True,
        drop_last=True, 
        collate_fn=train_dataset.yolo_dataset_collate,
        sampler=train_sampler
    )
    
    val_dataset = YoloDataset(
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
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset, 
        shuffle = False, 
        batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
        num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
        pin_memory=True,
        drop_last=True, 
        collate_fn=val_dataset.yolo_dataset_collate,
        sampler=val_sampler
    )



        
    for epoch in range(CONFIGS.TRAIN_START_EPOCHS, CONFIGS.TRAIN_EPOCHS):
        
        # train epoch
        for batch_idx, batch in enumerate(train_loader):
            images, targets, y_trues = batch[0], batch[1], batch[2]

            outputs = ddp_model(images.to(local_rank))
            loss_all  = 0
            losses = [0, 0, 0]
            for l, output in enumerate(outputs):
                loss_item = criterion(
                    l,
                    output,
                    targets,
                    y_trues[l]
                )
                loss_all  += loss_item
                losses[l] += loss_item
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                sllog << f'[TRAIN][{epoch}][{local_rank}][{batch_idx}]  loss: {loss_all.item():.4f}  loss-1: {losses[0]:.4f}  loss-2: {losses[1]:.4f}  loss-3: {losses[2]:.4f}'
    
        # save model
        if local_rank == 0:
            torch.save(ddp_model, f'/data/ylw/code/pl_yolo_v5/checkpoints/{epoch}.pth')

        # eval
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, targets, y_trues = batch[0], batch[1], batch[2]

                outputs = ddp_model(images)
                loss_all  = 0
                losses = [0, 0, 0]
                for l, output in enumerate(outputs):
                    loss_item = criterion(
                        l,
                        output,
                        targets,
                        y_trues[l]
                    )
                    loss_all  += loss_item
                    losses[l] += loss_item.item()

                if batch_idx % 10 == 0:
                    sllog << f'[EVAL][{epoch}][{local_rank}][{batch_idx}]  loss: {loss_all.item():.4f}  loss-1: {losses[0]:.4f}  loss-2: {losses[1]:.4f}  loss-3: {losses[2]:.4f}'

    # test


def main():
    worker_size = 2
    mp.spawn(run_worker,
        args=(worker_size,),
        nprocs=worker_size,
        join=True)



if __name__=="__main__":
    main()
