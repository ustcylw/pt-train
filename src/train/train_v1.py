# ##################################################
#
# ##################################################
#! /usr/bin/env python
# coding: utf-8
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


class TrainCallback(GraphCallback):
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # sllog << f'{len(batch)=}  {batch[0].shape=}  {batch[0].device=}  {pl_module.data_module.train_loader.dataset.input_shape=}'
        self.add_scalar(
            tag='img_size',
            scalar_value=pl_module.data_module.train_dataset.input_shape[0],
            global_step=trainer.global_step
        )


class YoloV5DataModule(LightningDataModule):
    name = 'pl-yolo-v5'
    def __init__(
        self, 
        configs,
        val_split: float = 0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.configs = configs
        self.val_split = val_split
        self.args = args
        self.kwargs = kwargs

    @property
    def num_classes(self):
        return self.configs.VOC_CLS_NUM

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        """Split the train and valid dataset."""
        # extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        # dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        # train_length = len(dataset)
        # self.dataset_train, self.dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        ...

    def train_dataloader(self):
        
        train_annotation_path = os.path.join(self.configs.DATASET_DIR, '2007_train.txt')
        with open(train_annotation_path, encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[:int(len(lines)*(1-self.val_split))]
            
        self.train_dataset = YoloDataset(
            lines, 
            (self.configs.TRAIN_IMG_SIZE, self.configs.TRAIN_IMG_SIZE), 
            self.configs.VOC_CLS_NUM, 
            np.array(self.configs.ANCHORS), 
            self.configs.ANCHORS_MASK, 
            mosaic=self.configs.DATASET_MOSAIC, 
            mixup=self.configs.DATASET_MIXUP, 
            mosaic_prob=self.configs.DATASET_MOSAIC_PROB, 
            mixup_prob=self.configs.DATASET_MIXUP_PROB, 
            train=True,
            fixed_input_shape=False
        )

        self.train_loader = DataLoader(
            self.train_dataset, 
            shuffle = True, 
            batch_size = self.configs.TRAIN_BATCH_SIZE, 
            num_workers = self.configs.TRAIN_NUMBER_WORKERS, 
            pin_memory=True,
            drop_last=True, 
            collate_fn=self.train_dataset.yolo_dataset_collate
        )
        
        return self.train_loader

    def val_dataloader(self):
        train_annotation_path = os.path.join(self.configs.DATASET_DIR, '2007_train.txt')
        with open(train_annotation_path, encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[int(len(lines)*(1-self.val_split)):]
            
        self.val_dataset = YoloDataset(
            lines, 
            (self.configs.TRAIN_IMG_SIZE, self.configs.TRAIN_IMG_SIZE), 
            self.configs.VOC_CLS_NUM, 
            np.array(self.configs.ANCHORS), 
            self.configs.ANCHORS_MASK, 
            mosaic=self.configs.DATASET_MOSAIC, 
            mixup=self.configs.DATASET_MIXUP, 
            mosaic_prob=self.configs.DATASET_MOSAIC_PROB, 
            mixup_prob=self.configs.DATASET_MIXUP_PROB, 
            train=False,
            fixed_input_shape=True
        )

        self.val_loader = DataLoader(
            self.val_dataset, 
            shuffle = False, 
            batch_size = self.configs.TRAIN_BATCH_SIZE, 
            num_workers = self.configs.TRAIN_NUMBER_WORKERS, 
            pin_memory=True,
            drop_last=True, 
            collate_fn=self.val_dataset.yolo_dataset_collate
        )
        
        return self.val_loader
        
    def test_dataloader(self):
        
        test_annotation_path = os.path.join(self.configs.DATASET_DIR, '2007_val.txt')
        with open(test_annotation_path, encoding='utf-8') as f:
            lines = f.readlines()
            
        self.test_dataset = YoloDataset(
            lines, 
            (self.configs.TRAIN_IMG_SIZE, self.configs.TRAIN_IMG_SIZE), 
            self.configs.VOC_CLS_NUM, 
            np.array(self.configs.ANCHORS), 
            self.configs.ANCHORS_MASK, 
            mosaic=self.configs.DATASET_MOSAIC, 
            mixup=self.configs.DATASET_MIXUP, 
            mosaic_prob=self.configs.DATASET_MOSAIC_PROB, 
            mixup_prob=self.configs.DATASET_MIXUP_PROB, 
            train=False,
            fixed_input_shape=True
        )

        self.test_loader = DataLoader(
            self.test_dataset, 
            shuffle = False, 
            batch_size = self.configs.TRAIN_BATCH_SIZE, 
            num_workers = self.configs.TRAIN_NUMBER_WORKERS, 
            pin_memory=False,
            drop_last=True, 
            collate_fn=self.test_dataset.yolo_dataset_collate
        )

        return self.test_loader


class YoloV5Trainer(PLMInterface):
    
    def __init__(self, hparams=...):
        super().__init__(hparams)

        self.model = self.create_model()
        self.criterion = YoloLoss.YOLOLoss(
            np.array(self.hparams.configs.ANCHORS),
            self.hparams.configs.VOC_CLS_NUM,
            (
                self.hparams.configs.TRAIN_IMG_SIZE,
                self.hparams.configs.TRAIN_IMG_SIZE
            ),
            self.hparams.configs.ANCHORS_MASK,
            label_smoothing=0
        )

        sllog << f'='*80
        TISummary(self.model)
        sllog << f'-'*80

        # self.writer = SummaryWriter(logdir=f'{self.hparams.configs.ROOT_DIR}/runs')
        # input_dummy = torch.zeros(size=(1, 3, self.hparams.configs.TRAIN_IMG_SIZE, self.hparams.configs.TRAIN_IMG_SIZE)).to(self.device)
        # self.writer.add_graph(self.model, input_to_model=input_dummy)
        
        self.mAP = 0

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

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self.forward(images)
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(l, output, targets, y_trues[l])
            loss_all  += loss_item
            losses[l] += loss_item

        return {'loss': loss_all, 'layer-1': losses[0], 'layer-2': losses[1], 'layer-3': losses[2]}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, y_trues = batch[0], batch[1], batch[2]

        outputs = self(images)
        loss_all  = 0
        losses = [0, 0, 0]
        for l, output in enumerate(outputs):
            loss_item = self.criterion(l, output, targets, y_trues[l])
            loss_all  += loss_item
            losses[l] += loss_item

        # sllog << f'[============]  {images.shape=}  {loss_all=}'

        return {
            'val_loss': loss_all.detach().cpu().item(), 
            'val-layer-1': losses[0].detach().cpu().item(), 
            'val-layer-2': losses[1].detach().cpu().item(), 
            'val-layer-3': losses[2].detach().cpu().item()
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = batch
        # p, p_d = self(imgs)
        # return p, p_d, batch, batch_idx
        ...

    def training_step_end(self, step_output):
        # self.log('train_loss', step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('lr', self.lr_schedulers().get_lr(), step_output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # torch.cuda.empty_cache()
        ...
        
    def validation_step_end(self, step_output):
        # self.log("val_loss", loss.item(), prog_bar=True, logger=True)
        # self.log("val_giou_loss", loss_giou.item(), prog_bar=True, logger=True)
        # self.log("val_conf_loss", loss_conf.item(), prog_bar=True, logger=True)
        # self.log("val_cls_loss", loss_cls.item(), prog_bar=True, logger=True)
        # torch.cuda.empty_cache()
        # return step_output
        ...
        
    def training_epoch_end(self, outputs):
        # ret = super().training_epoch_end(outputs)
        # if self.current_epoch > 5:
        #     print('*'*20+"Validate"+'*'*20)
        #     with torch.no_grad():
        #         APs = Evaluator(self.yolov3).APs_voc()
        #         for i in APs:
        #             print("{} --> mAP : {}".format(i, APs[i]))
        #             mAP += APs[i]
        #         mAP = mAP / self.train_dataset.num_classes
        #         print('mAP:%g'%(mAP))
                
        #     # self.writer.add_scalars(main_tag='val', tag_scalar_dict=APs+{'map':mAP})
        # return ret
        ...
    
    def validation_epoch_end(self, outputs):
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        self.log("metric_loss", np.array([output['val_loss'] for output in outputs]).mean(), prog_bar=True, logger=True)
        
    def test_step_end(self, *args, **kwargs):

        # labels = [f'{self.hparams.configs.VOC_CLASSES[int(cls)]} {score}' for cls, score in zip(rets[:, 4], rets[:, 5])]
        # image = CVDraw.rectangles(image=cv2.UMat(image.astype(np.uint8)), bboxes=rets, labels=labels)
        # if self.hparams.configs.TEST_SHOW:
        #     CVDraw.Show(name='cv-image-1').show(image=image, wait_time=100)
        # if self.hparams.configs.TEST_SAVE:
        #     image_file = os.path.join(self.hparams.configs.TEST_SAVE_DIR, f'{batch_idx}-{i}.jpg')
        #     CVDraw.save_image(image=image, image_file=image_file)
        ...
        
    def test_epoch_end(self, outputs):
        ...

    def predict_step(self, batch, batch_idx):
        
        # preds = self(batch[0])
        # preds = preds.cpu()

        # images = batch[0].cpu().numpy()
        # batch[1] = batch[1].cpu()
        
        # h, w, _ = CONFIGS.INPUT_SHAPE

        # for i in range(preds.shape[0]):
            
        #     pred = preds[i, :, :, :]
        #     bboxes, cls_indexs, probs = decoder(pred)

        #     image = images[i, :, :, :]
        #     image = image.transpose((1, 2 ,0)) * 255

        #     result = []
        #     clses = []
        #     for i,box in enumerate(bboxes):
        #         x1 = int(box[0]*w)
        #         x2 = int(box[2]*w)
        #         y1 = int(box[1]*h)
        #         y2 = int(box[3]*h)
        #         cls_index = cls_indexs[i]
        #         cls_index = int(cls_index) # convert LongTensor to int
        #         prob = probs[i]
        #         prob = float(prob)
        #         result.append([x1, y1, x2, y2, cls_index,prob])
        #         clses.append(CONFIGS.VOC_CLASSES[cls_index])

        #     torch.cuda.empty_cache()
            
        #     rets = BBoxes(result, mode='xyxycs')
        #     if len(rets.shape) < 2:
        #         continue
        #     if rets.shape[0] <= 0:
        #         continue
            
        #     labels = [f'{CONFIGS.VOC_CLASSES[int(cls)]} {score}' for cls, score in zip(rets[:, 4], rets[:, 5])]
        #     image = CVDraw.rectangles(image=cv2.UMat(image.astype(np.uint8)), bboxes=rets, labels=labels)
        #     if CONFIGS.TEST_SHOW:
        #         CVDraw.Show(name='cv-image-1').show(image=image, wait_time=100)
        #     if CONFIGS.TEST_SAVE:
        #         image_file = os.path.join(CONFIGS.TEST_SAVE_DIR, f'{batch_idx}-{i}.jpg')
        #         print(f'{image_file=}  {self.device}')
        #         CVDraw.save_image(image=image, image_file=image_file)
        ...

    def configure_optimizers(self):
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.hparams.configs.TRAIN_LR_INIT,
            momentum=self.hparams.configs.TRAIN_MOMENTUM, 
            weight_decay=self.hparams.configs.TRAIN_WEIGHT_DECAY
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                # "scheduler": cosine_lr_scheduler.CosineDecayLR(
                #     self.optimizer,
                #     T_max=10000,  # self.hparams.configs.TRAIN_EPOCHS*len(self.train_dataloader()),
                #     lr_init=self.hparams.configs.TRAIN_LR_INIT,
                #     lr_min=self.hparams.configs.TRAIN_LR_END,
                #     warmup=200, # self.hparams.configs.TRAIN_WARMUP_EPOCHS*len(self.train_dataloader())
                # ),
                'scheduler': ReduceLROnPlateau(
                    self.optimizer, 
                    factor=0.6, 
                    patience=2, 
                    verbose=True, 
                    mode="min", 
                    threshold=1e-3, 
                    min_lr=1e-8, 
                    eps=1e-8
                ), 
                "monitor": "metric_loss",
                "frequency": 1  # "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    
    def configure_callbacks(self):
        log_dir = os.path.join(self.hparams.configs.ROOT_DIR, f'checkpoints/{self.hparams.configs.PRE_SYMBOL}_{self.hparams.configs.POST_SYMBOL}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_checkpoint = PLMI.ModelCheckpoint(
            monitor='metric_loss',
            dirpath=log_dir,
            filename='{epoch}-{metric_loss:.2f}',
            save_top_k=-1,
            save_last=True,
            every_n_epochs=1
        )
        log_dir = os.path.join(self.hparams.configs.ROOT_DIR, f'runs/{self.hparams.configs.PRE_SYMBOL}_{self.hparams.configs.POST_SYMBOL}')
        graph_callback = TrainCallback(
            interval=self.hparams.configs.TRAIN_LOG_INTERVAL, 
            epoch_step=True, 
            log_dir=log_dir, 
            dummy_input=torch.rand(
                1, 
                3, 
                self.hparams.configs.TRAIN_IMG_SIZE,
                self.hparams.configs.TRAIN_IMG_SIZE
            )
        )
        return [model_checkpoint, graph_callback]


if __name__ == '__main__':
    

    dm = YoloV5DataModule(
        configs=CONFIGS,
        val_split = 0.2
    )
    
    model = YoloV5Trainer(
        hparams={
            'batch_size': CONFIGS.TRAIN_BATCH_SIZE,
            'auto_scale_batch_size': True,
            'auto_lr_find': True,
            'learning_rate': CONFIGS.TRAIN_LR_INIT,
            'reload_dataloaders_every_epoch': False,
            'hidden_dim': 32,
            'configs': CONFIGS
        }
    )
    model.set_data_module(data_module=dm)

    trainer = pl.Trainer(
        max_steps=-1,
        max_epochs=80, 
        accelerator="gpu", 
        devices=2,
        enable_checkpointing=True,
        weights_save_path=os.path.join(CONFIGS.ROOT_DIR, 'runs/checkpoints'),
        log_every_n_steps=CONFIGS.TRAIN_LOG_INTERVAL,
        precision=32,
        val_check_interval=1.0,
        limit_train_batches = 20,
        limit_val_batches = 10,
        # limit_test_batches = 1.0,
        # enable_model_summary=True
        # resume_from_checkpoint=os.path.join(CONFIGS.TRAIN_PRETRAINED_DIR, CONFIGS.TRAIN_PRETRAINED_MODEL_NAME)
    )

    dm.setup(CONFIGS.MODE)
    if CONFIGS.MODE == 'train':
        trainer.fit(model, datamodule=dm)
    elif CONFIGS.MODE == 'test':
        # model = PLYOLOV1Trainer.load_from_checkpoint(CONFIGS.PRETRAINED)
        model = model.load_from_checkpoint(
            CONFIGS.TEST_PRETRAINED_MODEL_NAME, 
            hparams_file=f'/data/ylw/code/pl_yolo_v1/lightning_logs/version_238/hparams.yaml'
        )
        model.eval()
        trainer.testing = True
        trainer.test(model, datamodule=dm)
    elif CONFIGS.MODE == 'predict':
        
        # val_dataset = yoloDataset(
        #         list_file='voc2007test.txt',
        #         train=False,
        #         transform=[transforms.ToTensor()],
        #         config=CONFIGS
        #     )

        # val_loader = DataLoader(
        #         val_dataset,
        #         batch_size=CONFIGS.BATCH_SIZE,
        #         shuffle=False,
        #         drop_last=True,
        #         pin_memory=False,
        # )

        # map_location = {'cpu': 'cuda:0'}  # 好像不起作用
        # model = PLT.PLYOLOV1Trainer.load_from_checkpoint(
        #     CONFIGS.PRETRAINED, 
        #     map_location=map_location,
        #     hparams_file=f'/data/ylw/code/git_yolo/pytorch-YOLO-v1/pl_yolo_v1/lightning_logs/version_90/hparams.yaml'
        # )
        # print(f'{model.device=}')
        # model.eval()
        # model.freeze()
        # print(f'{model.device=}')
        # model.to('cuda:0')
        # print(f'{model.device=}')

        # trainer = pl.Trainer(
        #     accelerator="gpu", 
        #     devices=1,
        #     precision=16,
        # )
        
        # trainer.predict(model=model, dataloaders=val_loader)
        ...
