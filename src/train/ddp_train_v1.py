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




class Trainer():
    
    def __init__(self, start_epoch, epochs, *args, **kwargs):
        #  环境信息
        #  训练信息
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.data_module = None
        self.current_epoch = start_epoch
        self.current_batch_idx = 0
        self.global_batch_idx = 0
        self.val_interval: Union[int, float] = 1
        self.acc_grad = 1
        #  验证信息
        #  测试信息
        #  策略信息
        self.hparams = {}
        self.optims = self.configure_optimizers()
        self.cbs = self.configure_callback()
        self.rank = -1
        self.world_size = -1
    
    def setup(self, rank, world_size):
        '''
        配置环境
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        
        # init params
        if rank == 0:
            self.get_model()
        self._fit(self.model, self.data_module)

    def cleanup(self, ):
        dist.destroy_process_group()

    # #############################################################
    # 外部调用接口
    # #############################################################
    def fit(self, model, data_module):
        self.model = model
        self.data_module = data_module
        mp.spawn(self.setup,
                    args=(1,),
                    nprocs=2,
                    join=True)

    def _fit(self, model, data_module):
        self.run_fit()
    
    def test(self, model, data_loader):
        ...

    def predict(self, image_list):
        ...
    
    # #############################################################
    # 内部调用接口
    # #############################################################
    def get_model(self, *args, **kwargs):
        ...
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, step_output):
        if isinstance(step_output, torch.Tensor):
            loss = step_output
        else:
            loss = step_output['loss']
            
        loss.backward()
        if self.acc_grad == 1 or (self.current_batch_idx+1) % self.acc_grad == 0:
            self.optims['optimizer'].step()
            self.optims['optimizer'].zeros_grad()

    def train_step(self, batch, batch_idx) -> Union[None, torch.Tensor, dict]:
        return None
    
    def train_step_end(self, step_output: Union[None, torch.Tensor, dict]) -> Union[None, torch.Tensor, dict]:
        if step_output is None:
            return
    
    def train_epoch(self, model, data_loader) -> Union[None, torch.Tensor, dict]:

        for batch_idx, batch in enumerate(data_loader):
            self.current_batch_idx = batch_idx
            self.global_batch_idx += 1
            
            step_output = self.train_step(batch, batch_idx)
            
            self.backward(step_output)
                
            self.train_step_end(step_output=step_output)

            # lr
        
        return None
    
    def train_epoch_end(self, epoch_output: Union[None, torch.Tensor, dict]) -> Union[None, torch.Tensor, dict]:
        if epoch_output is None:
            return
    
    def val_step(self, batch, batch_idx):
        ...
    
    def val_step_end(self, step_output):
        ...
    
    def run_fit(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
            self.current_epoch = epoch
            
            epoch_output = self.train_epoch(self.model, self.data_module)
            
            self.train_epoch_end(epoch_output)

        # lr
        # save params
        if self.cbs is not None and len(self.cbs) > 0:
            for cb in self.cbs:
                cb()
            
    def save_hparams(self, hparams):
        if 'hparams' not in self.__dict__:
            raise ValueError(f'There is no |hparams| param.')
        print(f'*'*80)
        self.__dict__['hparams'].update(hparams)

    def configure_optimizers(self):
        return {
            "optimizer": None,
            "lr_scheduler": None
        }

    def configure_callback(self):
        return {}


class MyTrainer(Trainer):
    
    def __init__(self, start_epoch, epochs, hparams=...):
        super().__init__(start_epoch, epochs, hparams)
        self.save_hparams(hparams)
        print(f'{self.hparams=}')
        # self.model = self.create_model()
        self.criterion = YoloLoss.YOLOLoss(
            np.array(self.hparams['configs'].ANCHORS),
            self.hparams['configs'].VOC_CLS_NUM,
            (
                self.hparams['configs'].TRAIN_IMG_SIZE,
                self.hparams['configs'].TRAIN_IMG_SIZE
            ),
            self.hparams['configs'].ANCHORS_MASK,
            label_smoothing=0
        )

        # sllog << f'='*80
        # TISummary(self.model)
        # sllog << f'-'*80

    def create_model(self):
        sllog << f'[create_model] loading model ...'

        phi = 's'
        backbone        = 'cspdarknet'
        pretrained      = False

        model = YoloBody(
            self.hparams.configs.ANCHORS_MASK, 
            self.hparams.configs.VOC_CLS_NUM, 
            phi, 
            backbone, 
            pretrained=pretrained, 
            input_shape=(self.hparams.configs.TRAIN_IMG_SIZE, self.hparams.configs.TRAIN_IMG_SIZE)
        )
        
        sllog << f'[create_model] load model complete.'
    
        return model

    def train_step(self, batch, batch_idx) -> Union[None, torch.Tensor, dict]:
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self.forward(images)
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(l, output, targets, y_trues[l])
            loss_all  += loss_item
            losses[l] += loss_item

        return {'loss': loss_all, 'layer-1': losses[0], 'layer-2': losses[1], 'layer-3': losses[2]}
    
    def train_step_end(self, step_output: Union[None, torch.Tensor, dict]) -> Union[None, torch.Tensor, dict]:
        if step_output is None:
            return
        for k,v in step_output.items():
            sllog << f'[train_step_end]  {k}={v.detatch().cpu().item():.4f}'
    
    def train_epoch_end(self, epoch_output: Union[None, torch.Tensor, dict]) -> Union[None, torch.Tensor, dict]:
        if epoch_output is None:
            return
        sllog << f'[train_epoch_end]'



class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    import numpy as np
    
    

    train_annotation_path = os.path.join(CONFIGS.DATASET_DIR, '2007_train.txt')
    with open(train_annotation_path, encoding='utf-8') as f:
        lines = f.readlines()
        
    train_dataset = YoloDataset(
        lines, 
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

    train_loader = DataLoader(
        train_dataset, 
        shuffle = True, 
        batch_size = CONFIGS.TRAIN_BATCH_SIZE, 
        num_workers = CONFIGS.TRAIN_NUMBER_WORKERS, 
        pin_memory=True,
        drop_last=True, 
        collate_fn=train_dataset.yolo_dataset_collate
    )
    
    phi = 's'
    backbone        = 'cspdarknet'
    pretrained      = False

    model = YoloBody(
        CONFIGS.ANCHORS_MASK, 
        CONFIGS.VOC_CLS_NUM, 
        phi, 
        backbone, 
        pretrained=pretrained, 
        input_shape=(CONFIGS.TRAIN_IMG_SIZE, CONFIGS.TRAIN_IMG_SIZE)
    )
        
    t = MyTrainer(start_epoch=0, epochs=3, hparams={'configs': CONFIGS})
    
    t.fit(model, train_loader)
    
    
    
    # n_gpus = torch.cuda.device_count()
    # if n_gpus < 2:
    #     print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    # else:
    #     run_demo(demo_basic, 2)
    #     # run_demo(demo_checkpoint, 2)
    #     # run_demo(demo_model_parallel, 2)