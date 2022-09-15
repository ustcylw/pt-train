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


class TrainModule():

    def __init__(self, pretrained=None):
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

        pretrained = f'/data/ylw/code/pl_yolo_v5/checkpoints/400.pth'
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
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=CONFIGS.TRAIN_LR_INIT,
            momentum=CONFIGS.TRAIN_MOMENTUM, 
            weight_decay=CONFIGS.TRAIN_WEIGHT_DECAY
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
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=CONFIGS.TRAIN_LR_END)

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
                y_trues[l]
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


class Callback():
    def __init__(self):
        ...

    def on_init_start(self, local_rank):
        ...

    def on_init_end(self, local_rank, module:TrainModule):
        ...

    def on_train_start(self, local_rank, module:TrainModule):
        ...

    def on_train_end(self, local_rank, module:TrainModule):
        ...

    def on_train_epoch_start(self, local_rank, epoch, module:TrainModule):
        ...

    def on_train_epoch_end(self, local_rank, epoch, module:TrainModule):
        ...

    def on_train_step_start(self, local_rank, module:TrainModule, epoch, batch_idx, batch, global_step):
        ...

    def on_train_step_end(self, local_rank, module:TrainModule, epoch, batch_idx, batch, step_outputs, global_step):
        ...
        
    def on_eval_step_start(self, local_rank, module:TrainModule, batch_idx, batch, global_step):
        ...

    def on_eval_step_end(self, local_rank, module:TrainModule, batch_idx, batch, step_outputs, global_step):
        ...
        
    def on_test_step_start(self, local_rank, module:TrainModule, batch_idx, batch):
        ...

    def on_test_step_end(self, local_rank, module:TrainModule, batch_idx, batch, step_outputs):
        ...


class LogCallback(Callback):
    def __init__(self, meters={}, log_dir=None, log_surfix='default'):
        super(LogCallback, self).__init__()
        self.meters = {
            'fps': AverageMeter('fps', fmt=':3i'),
            'elapse': AverageMeter('elapse', fmt=':4f')
        }
        self.meters.update(meters)
        self.step_start_time = 0
        self.step_end_time = 0
        self.log_dir = log_dir
        if self.log_dir is None:
            self.log_dir = './'
        self.log_surfix = log_surfix

        self.LOG = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # consol_fmt = '[%(asctime)-25s] [%(filename)s-20:%(lineno)d] %(levelname)s: %(message)s'
        # consol_formatter = logging.Formatter(consol_fmt)  # 设置日志格式
        file_fmt = '[%(asctime)-25s] [%(filename)s-20:%(lineno)d] %(levelname)s: %(message)s'
        file_formatter = logging.Formatter(file_fmt)  # 设置日志格式
        # print(f'='*80)        
        # if True: # log_stream in [1, 2]:
        #     sh = logging.StreamHandler(stream=sys.stderr)  # 往屏幕上输出
        #     sh.setFormatter(consol_formatter)  # 设置屏幕上显示的格式
        #     self.LOG.addHandler(sh)  # 把对象加到logger里
        if True: # log_stream in [0, 2]:
            str_date = time.strftime(
                '"%Y-%m-%d-%H-%M-%S"', time.localtime(int(time.time())))
            file_name = os.path.join(self.log_dir, f'{str_date[1:-1]}-{self.log_surfix}.log')
            th = handlers.RotatingFileHandler(filename=file_name, maxBytes=10*1024*1024, backupCount=3)
            th.setFormatter(file_formatter)
            self.LOG.addHandler(th)


    def on_init_start(self, local_rank):
        sllog << f'[on_init_start]  {local_rank=}'

    def on_init_end(self, local_rank, module:TrainModule):
        sllog << f'[on_init_end]  {local_rank=}'

    def on_train_start(self, local_rank, module:TrainModule):
        ...

    def on_train_end(self, local_rank, module:TrainModule):
        ...

    def on_train_epoch_start(self, local_rank, epoch, module:TrainModule):
        sllog << f'[on_train_epoch_start]  {local_rank=}  {epoch=}'

    def on_train_epoch_end(self, local_rank, epoch, module:TrainModule):
        sllog << f'[on_train_epoch_end]  {local_rank=}  {epoch=}'

    def on_train_step_start(self, local_rank, module:TrainModule, epoch, batch_idx, batch, global_step):
        if local_rank == 0:
            self.step_start_time = time.time()
            # self.step_start_time = datetime.now()

    def on_train_step_end(self, local_rank, module:TrainModule, epoch, batch_idx, batch, step_outputs, global_step):
        if local_rank == 0 and batch_idx % 30 == 0:
            self.step_end_time = time.time()
            # self.step_end_time = datetime.now()
            self.meters['fps'].update(val=int(batch[0].shape[0]), n=self.step_start_time-self.step_end_time)
            self.meters['elapse'].update(val=self.step_end_time-self.step_start_time, n=1)
            # self.meters['elapse'].update(val=float((self.step_end_time-self.step_start_time).seconds), n=1)

            if len(step_outputs) > 0:
                for k, v in step_outputs.items():
                    self.meters[k].update(val=v.item(), n=batch[0].shape[0])

            table=PrettyTable(list(self.meters.keys()))
            table.float_format = '.8'
            table.int_format = '3'
            table.add_row([meter.avg if meter.name != "elapse" else meter.sum for _, meter in self.meters.items()])
            # sllog << f"\n{table.get_string()}"
            self.LOG.info(f"\n{table.get_string()}")

    def on_eval_step_start(self, local_rank, module:TrainModule, batch_idx, batch, global_step):
        if local_rank == 0:
            self.step_start_time = time.time()

    def on_eval_step_end(self, local_rank, module:TrainModule, batch_idx, batch, step_outputs, global_step):
        if local_rank == 0 and batch_idx % 10 == 0:
            loss_all, loss1, loss2, loss3 = step_outputs['loss'], step_outputs['loss1'], step_outputs['loss2'], step_outputs['loss3']
            sllog << f'[EVAL][{local_rank}][{batch_idx}]  loss: {loss_all.item():.4f}  loss-1: {loss1:.4f}  loss-2: {loss2:.4f}  loss-3: {loss3:.4f}'

    def on_test_step_start(self, local_rank, module:TrainModule, batch_idx, batch):
        ...

    def on_test_step_end(self, local_rank, module:TrainModule, batch_idx, batch, step_outputs):
        if local_rank == 0 and batch_idx % 10 == 0:
            loss_all, loss1, loss2, loss3 = step_outputs['loss'], step_outputs['loss1'], step_outputs['loss2'], step_outputs['loss3']
            sllog << f'[TEST][{local_rank}][{batch_idx}]  loss: {loss_all.item():.4f}  loss-1: {loss1:.4f}  loss-2: {loss2:.4f}  loss-3: {loss3:.4f}'


class GraphCallback(Callback, TBWriter.SummaryWriter):
    '''
        可以不用繼承TBWriter.SummaryWriter
    '''
    def __init__(
        self,
        interval=1,
        log_dir='./',
        dummy_input=None,
        *args,
        **kwargs
    ):
        # super(GraphCallback, self).__init__(interval=interval, epoch_step=epoch_step, *args, **kwargs)
        Callback.__init__(self)
        TBWriter.SummaryWriter.__init__(self, log_dir=log_dir, flush_secs=30)
        self.dummy_input = torch.FloatTensor(dummy_input)
        self.log_dir = log_dir
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        
    def on_train_start(self, local_rank, module:TrainModule) -> None:
        super().on_train_start(local_rank, module)
        
        if local_rank == 0 and self.dummy_input is not None:
            # self.logger = TBLogger(name=None, log_surfix='default', log_dir=self.log_dir, backup_count=3, max_bytes=5, log_stream=0, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='')
            # self.logger.add_graph(pl_module.model, (self.dummy_input.to(local_rank),))
            self.add_graph(module.model, (self.dummy_input.to(local_rank),))
            self.dummy_input = self.dummy_input.cpu().numpy()

    def on_train_step_end(
        self,
        local_rank,
        module:TrainModule,
        epoch,
        batch_idx,
        batch,
        step_outputs,
        global_step
    ) -> None:
        if (batch_idx+1) % self.interval == 0:
            for tag, value in module.model.named_parameters():
                # self.add_histogram(tag, value.detach().cpu().numpy(), trainer.global_step)
                self.add_histogram(tag, value, global_step)
            if isinstance(step_outputs, torch.Tensor):
                self.add_scalar(tag='train_loss', scalar_value=step_outputs.detach().cpu().item(), global_step=global_step)
            else:
                scalars_dict = {k: v.detach().cpu().item() for k, v in step_outputs.items()}
                self.add_scalars(main_tag='TRAIN-LOSS', tag_scalar_dict=scalars_dict, global_step=global_step)
                self.add_scalar(
                    tag='lr',
                    scalar_value=module.optimizer.param_groups[0]['lr'],
                    global_step=global_step
                )

    def on_eval_step_end(
        self, 
        local_rank,
        module:TrainModule,
        batch_idx,
        batch,
        step_outputs,
        global_step
    ) -> None:
        if isinstance(step_outputs, torch.Tensor):
            self.add_scalar(tag='val_loss', scalar_value=step_outputs.detach().cpu().item(), global_step=global_step)
        else:
            scalars_dict = {k: v.detach().cpu().item() for k, v in step_outputs.items()}
            self.add_scalars(main_tag='VAL-LOSS', tag_scalar_dict=scalars_dict, global_step=global_step)


class TrainCallback(GraphCallback):
    
    def __init__(self, interval=1, log_dir='./', dummy_input=None, *args, **kwargs):
        super().__init__(interval, log_dir, dummy_input, *args, **kwargs)
        
    def on_train_step_end(
        self,
        local_rank,
        module:TrainModule,
        epoch,
        batch_idx,
        batch,
        step_outputs,
        global_step
    ):
        super().on_train_step_end(local_rank, module, epoch, batch_idx, batch, step_outputs, global_step)
        # sllog << f'{len(batch)=}  {batch[0].shape=}  {batch[0].device=}  {pl_module.data_module.train_loader.dataset.input_shape=}'
        if (batch_idx+1) % self.interval == 0:
            self.add_scalar(
                tag='img_size',
                scalar_value=batch[0].shape[2],
                global_step=global_step
            )


class Trainer():

    def __init__(
        self, 
        train_module, 
        mode='train', 
        accelarate=1, 
        precision=False, 
        grad_clip=False,
        grad_max_norm=2e-2,
        seed=None, 
        grad_average=False,
        sync_bn=False
    ):
        self.train_module = train_module
        self.mode = mode
        self.start_epoch = 0
        self.num_epochs = 100
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.global_step_eval = 0
        self.local_rank = 0
        self.accelarate = accelarate
        self.callbacks = None
        self.precision = precision
        self.grad_clip = grad_clip
        self.grad_max_norm = grad_max_norm
        self.seed = seed
        self.grad_average=grad_average
        self.sync_bn=sync_bn

    def set_env(self, local_rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['NCCL_DEBUG'] = "INFO"
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
    
        # create default process group
        dist.init_process_group(
            backend="nccl",
            rank=local_rank,
            world_size=world_size
        )
        
        self.local_rank = local_rank
        torch.cuda.set_device(self.local_rank)

        if self.precision:
            self.grid_scaler = GradScaler()
    
    def run_loop(self, local_rank, world_size):

        self.set_env(local_rank, world_size)

        self.callbacks = self.train_module.set_callbacks()
        if self.callbacks is None or len(self.callbacks) == 0:
            self.callbacks = [LogCallback()]

        for cb in self.callbacks:
            cb.on_init_start(local_rank)

        self.train_module.create_model(local_rank)

        # construct DDP model
        ddp_model = DDP(self.train_module.model, device_ids=[local_rank])
        self.train_module.model = ddp_model

        if self.sync_bn:
            # https://blog.csdn.net/zmm__/article/details/126034359
            # https://blog.csdn.net/qq_39967751/article/details/123382981
            self.train_module.model = self.sync_batchnorm(self.train_module.model)

        # define loss function and optimizer
        self.train_module.create_loss()
        self.train_module.create_optim(ddp_model)

        # define data loader
        self.train_module.create_data_loader()

        for cb in self.callbacks:
            cb.on_init_end(local_rank, self.train_module)

        if self.mode == 'train':
            # for cb in self.callbacks:
            #     cb.on_train_start(local_rank, self.train_module)
            for epoch in range(CONFIGS.TRAIN_START_EPOCHS, CONFIGS.TRAIN_START_EPOCHS+CONFIGS.TRAIN_EPOCHS):

                self.current_epoch = epoch + 1

                for cb in self.callbacks:
                    cb.on_train_epoch_start(local_rank, self.current_epoch, self.train_module)

                start_epoch_time = time.time()
                # train epoch
                self.train_module.model.train()
                
                with logging_redirect_tqdm():
                    for batch_idx, batch in enumerate(tqdm(self.train_module.train_loader)):

                        self.current_step = batch_idx + 1
                        self.global_step += 1

                        for cb in self.callbacks:
                            cb.on_train_step_start(local_rank, self.train_module, self.current_epoch, batch_idx, batch, self.global_step)

                        if self.precision:
                            with autocast():
                                step_outputs = self.train_module.train_step(batch_idx, batch, local_rank)

                            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                            # Backward passes under autocast are not recommended.
                            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                            self.grid_scaler.scale(step_outputs['loss']).backward()

                            if self.grad_average:
                                #  https://www.cnpython.com/qa/887326
                                self.average_gradients(self.train_module.model)
                                dist.barrier()

                            if self.current_step % self.accelarate == 0:

                                # torch.nn.utils.clip_grad_norm_(
                                #     self.train_module.model.parameters(), 
                                #     self.grad_max_norm
                                # )

                                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                                # otherwise, optimizer.step() is skipped.
                                self.grid_scaler.step(self.train_module.optimizer)

                                # Updates the scale for next iteration.
                                self.grid_scaler.update()
                                self.train_module.optimizer.zero_grad()
                        else:
                            step_outputs = self.train_module.train_step(batch_idx, batch, local_rank)

                            step_outputs['loss'].backward()

                            if self.grad_average:
                                self.average_gradients(self.train_module.model)
                                dist.barrier()

                            if self.current_step % self.accelarate == 0:

                                # torch.nn.utils.clip_grad_norm_(
                                #     self.train_module.model.parameters(), 
                                #     self.grad_max_norm
                                # )

                                self.train_module.optimizer.step()
                                self.train_module.optimizer.zero_grad()
                            
                        self.train_module.train_step_end(step_outputs, batch_idx, batch, local_rank)

                        dist.barrier()

                        for cb in self.callbacks:
                            cb.on_train_step_end(local_rank, self.train_module, self.current_epoch, batch_idx, batch, step_outputs, self.global_step)

                self.train_module.lr_scheduler.step()

                # eval
                self.train_module.model.eval()
                if self.train_module.val_loader is not None:
                    for batch_idx, batch in enumerate(self.train_module.val_loader):

                        self.global_step_eval += 1
                        
                        for cb in self.callbacks:
                            cb.on_eval_step_start(local_rank, self.train_module, batch_idx, batch, self.global_step)

                        step_outputs = self.train_module.eval_step(batch_idx, batch, local_rank)

                        self.train_module.eval_step_end(step_outputs, batch_idx, batch, local_rank)

                        for cb in self.callbacks:
                            cb.on_eval_step_end(local_rank, self.train_module, batch_idx, batch, step_outputs, self.global_step)

                        dist.barrier()

                # save model
                if local_rank == 0 and epoch % 1 == 0:
                    torch.save(self.train_module.model.module.state_dict(), f'/data/ylw/code/pl_yolo_v5/checkpoints/{epoch}.pth')

                end_epoch_time = time.time()
                sllog << f'[************]  {end_epoch_time - start_epoch_time}s'

            if local_rank == 0:
                torch.save(self.train_module.model.module.state_dict(), f'/data/ylw/code/pl_yolo_v5/checkpoints/last.pth')

        # test
        self.train_module.model.eval()
        if self.train_module.test_loader is not None:
            for batch_idx, batch in enumerate(self.train_module.val_loader):

                for cb in self.callbacks:
                    cb.on_test_step_start(local_rank, self.train_module, batch_idx, batch)

                step_outputs = self.train_module.test_step(batch_idx, batch, local_rank)

                self.train_module.test_step_end(step_outputs, batch_idx, batch, local_rank)

                for cb in self.callbacks:
                    cb.on_test_step_end(local_rank, self.train_module, batch_idx, batch, step_outputs)

                dist.barrier()

    def fit(self):
        worker_size = 2
        
        mp.spawn(trainer.run_loop,
            args=(worker_size,),
            nprocs=worker_size,
            join=True)

    def predict(self, images: Union[np.ndarray, torch.Tensor, list], device_id=0):
        '''
        图片预处理放在外边做，不同模型与处理方式不同。
        '''
        self.train_module.model.eval()
        with torch.no_grad():
            preds = self.train_module.predict(images)
            self.train_module.predict_end(preds)

    def average_gradients(self, model):
        """ Gradient averaging. """
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def sync_batchnorm(self, model):
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)



if __name__=="__main__":
    train_module = TrainModule()
    trainer = Trainer(
        train_module=train_module, 
        mode='train', 
        accelarate=1, 
        precision=False,
        grad_average=False,
        sync_bn=False
    )
    trainer.fit()
