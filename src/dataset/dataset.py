from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import random

from PyUtils.logs.print import *
from PyUtils.viz.cv_draw import *
from PyUtils.bbox import BBoxes
from PyUtils.utils.pre_process import norm
from PyUtils.utils.image import cvtColor, paste
import PyUtils.argus.image_argus as Argus
from PyUtils.utils.random import rand as Rand



class YoloDataset(Dataset):
    def __init__(
        self, 
        annotation_lines, 
        input_shape, 
        num_classes, 
        anchors, 
        anchors_mask, 
        mosaic, 
        mixup, 
        mosaic_prob, 
        mixup_prob, 
        train=True,
        fixed_input_shape=False
    ):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.fixed_input_shape  = fixed_input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4

    def set_input_shape(self, input_shape):
        if isinstance(input_shape, int):
            self.input_shape = (input_shape, input_shape)
        else:
            self.input_shape = input_shape
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and Rand() < self.mosaic_prob and self.train:
            lines = sample(self.annotation_lines, 3)  #  sample(seq, n) 从序列seq中选择n个随机且独立的元素
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            if self.mixup and Rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        # img_ = rectangles(cv2.UMat(image), BBoxes(box.copy(), mode='xyxy'), labels=[str(label) for label in box[:, 4]], copy=False)
        # save_image(img_, f'/data/ylw/code/pl_yolo_v5/test/rets/test_{np.random.randint(0, 1000)}.jpg')

        image       = np.transpose(norm(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)

        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        
        # h, w = image.shape[1:3]
        # img_ = (image.copy() * 255).astype(np.int8).transpose((1,2,0))
        # tmp = np.zeros_like(y_true[0][0, :, :, 0], dtype=np.float64)
        # tmp += y_true[0][0, :, :, 0] + y_true[0][0, :, :, 1] + y_true[0][1, :, :, 0] + y_true[0][1, :, :, 1] + y_true[0][2, :, :, 0] + y_true[0][2, :, :, 1]
        # tmp = cv2.resize(tmp, y_true[1][0, :, :, 0].shape)
        # tmp += y_true[1][0, :, :, 0] + y_true[1][0, :, :, 1] + y_true[1][1, :, :, 0] + y_true[1][1, :, :, 1] + y_true[1][2, :, :, 0] + y_true[1][2, :, :, 1]
        # tmp = cv2.resize(tmp, y_true[2][0, :, :, 0].shape)
        # tmp += y_true[2][0, :, :, 0] + y_true[2][0, :, :, 1] + y_true[2][1, :, :, 0] + y_true[2][1, :, :, 1] + y_true[2][2, :, :, 0] + y_true[2][2, :, :, 1]
        # tmp = cv2.resize(tmp, (h, w)).astype(np.uint8)
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        # save_image(img_, '/data/ylw/code/pl_yolo_v5/test/rets/test_1.jpg')
        # img_ = cv2.addWeighted(img_.astype(np.uint8), 0.5, tmp, 0.5, 1)
        # img_ = rectangles(cv2.UMat(img_), BBoxes(box.copy(), mode='xyxy'), copy=False)
        # save_image(img_, '/data/ylw/code/pl_yolo_v5/test/rets/test_2.jpg')
        # save_image(tmp, '/data/ylw/code/pl_yolo_v5/test/rets/test_3.jpg')

        return image, box, y_true

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = cv2.imread(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih, iw = image.shape[:2]
        h, w = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(np.float32,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.ones(shape=(h, w, 3)) * 128
            new_image = paste(new_image, image, x=dx, y=dy)
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box

        if iw > ih:
            pad = int((iw-ih)/2)
            image = np.pad(image, ((pad, pad), (0, 0), (0, 0)))
            box[:, (1, 3)] += pad
        else:
            pad = int((ih-iw)/2)
            image = np.pad(image, ((0, 0), (pad, pad), (0, 0)))
            box[:, (0, 2)] += pad

        scale_x, scale_y = w/image.shape[1], h/image.shape[0]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        box[:, (0, 2)] = box[:, (0, 2)] * scale_x
        box[:, (1, 3)] = box[:, (1, 3)] * scale_y

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = Rand()<.5
        if flip: 
            image = cv2.flip(image, 1)
            box[:, [0,2]] = w - box[:, [2,0]]

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape

        image_datas = [] 
        box_datas   = []
        for line_idx, line in enumerate(annotation_line):
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = cv2.imread(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            ih, iw = image.shape[:2]
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])

            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = Rand()<.5
            if flip and len(box)>0:
                image = cv2.flip(image, 1)
                box[:, [0,2]] = iw - box[:, [2,0]]

            image_datas.append(image)
            box_datas.append(box)

        new_image, new_bboxes = Argus.mosica(
            img_shape=(h, w, 3), 
            img_1=image_datas[0], bboxes1=BBoxes(np.array(box_datas[0]), mode='xyxy'),
            img_2=image_datas[1], bboxes2=BBoxes(np.array(box_datas[1]), mode='xyxy'),
            img_3=image_datas[2], bboxes3=BBoxes(np.array(box_datas[2]), mode='xyxy'),
            img_4=image_datas[3], bboxes4=BBoxes(np.array(box_datas[3]), mode='xyxy'),
        )

        new_image       = np.array(new_image, np.uint8)  # 拼接后图片
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        return new_image, new_bboxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)

        return new_image, new_boxes
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        # targets: <N, 5>  [x,y,w,h,c]  xywh in [0, 1]
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        #-----------------------------------------------------------#
        #   input_shape [640, 640]
        #-----------------------------------------------------------#
        input_shape = np.array(self.input_shape, dtype='int32')
        #-----------------------------------------------------------#
        #   grid_shapes： [[20, 20], [40, 40], [80, 80]]
        #-----------------------------------------------------------#
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        #-----------------------------------------------------------#
        #   y_true： <(3, 20, 20, 25), (3, 40, 40, 25), (3, 80, 80, 25)>
        #-----------------------------------------------------------#
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        #-----------------------------------------------------------#
        #   box_best_ratio <(3, 20, 20), (3, 40, 40), (3, 80, 80)>
        #-----------------------------------------------------------#
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        #-----------------------------------------------------------#
        #   如果当前图片没有obj，则target为空，即全是0。
        #-----------------------------------------------------------#
        if len(targets) == 0:
            return y_true
        
        #-----------------------------------------------------------#
        #   每层layer计算
        #-----------------------------------------------------------#
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            #-----------------------------------------------------------#
            #   获取当前层anchors
            #   [
            #     [[10,13], [16,30], [33,23]],
            #     [[30,61], [62,45], [59,119]],
            #     [[116,90], [156,198], [373,326]]
            #   ]
            #-----------------------------------------------------------#
            anchors = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]
            
            batch_target = np.zeros_like(targets)
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #   targets中xywh in [0,1]，缩放到当前层grid尺寸，即乘以当前层grid的hw。
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   ？？？
            #   这种根据<h1/h2,w1/w2, h2/h1,w1/w2>的方式进行ahcor选择，有点？？？
            #   wh                          : <num_true_box, 2>
            #   np.expand_dims(wh, 1)       : <num_true_box, 1, 2>
            #   anchors                     : <9, 2>，表示每层都有9个anchor
            #   np.expand_dims(anchors, 0)  : <1, 9, 2>
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : <num_true_box, 9, 2>
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的高宽的比值
            #   ratios_of_anchors_gt    : <num_true_box, 9, 2>
            #
            #   ratios                  : <num_true_box, 9, 4>
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : <num_true_box, 9>
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            max_ratios           = np.max(ratios, axis = -1)
            # sllog << f'[get_target]  {targets.shape=}  {anchors.shape=}  {ratios_of_gt_anchors.shape=}  {ratios_of_anchors_gt.shape=}  {ratios.shape=}  {max_ratios=} /   {max_ratios.shape=}'
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                #-------------------------------------------------------#
                #   anchor & anchors_mask位置对应
                #   self.anchors_mask[l]第l层的anchor位置掩码
                #   over_threshold[mask]第l层第k个掩码：
                #       如果该掩码为false，表示第l层的第t个box没有分配到该anchor；
                #       如果该掩码为true，表示第l层的第t个box分配到该anchor.
                #   ************第t个box可能分配给多个anchor。************
                #-------------------------------------------------------#
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #   从第t个box在当前层对应grid尺寸的grid位置(i,j)
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        #-------------------------------------------------------#
                        #   如果当前层(l)当前anchor(k)当前位置(local_j, local_i)已经
                        #   有一个bbox占据，此时就要比较是否是小于(ratio < )self.threshold
                        #   中最好的ratio的那个bbox，如果当前第l层第k个anchor的ratio大于新的ratio[mask]
                        #   则进行y_true的更新。
                        #-------------------------------------------------------#
                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        try:
                            y_true[l][k, local_j, local_i, c + 5] = 1
                        except Exception as e:
                            print(f'{e=}  {y_true[0].shape=}  {y_true[1].shape=}  {y_true[2].shape=}  {c=}  {c+5=}\n\n')
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
                        
        return y_true

    def yolo_dataset_collate(self, batch):
        '''
        DataLoader在init之后，就不能修改
        | batch_size, batch_sampler, sampler, drop_last, dataset, persistent_workers|
        这些内容，为了能够修改input_shape，collate作为dataset的一个属性函数。
        '''
        images  = []
        bboxes  = []
        y_trues = [[] for _ in batch[0][2]]
        for img, box, y_true in batch:
            images.append(img)
            bboxes.append(box)
            for i, sub_y_true in enumerate(y_true):
                y_trues[i].append(sub_y_true)
                
        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]

        if not self.fixed_input_shape:
            input_shape = random.choice([32*i for i in range(10, 30)])
            self.input_shape = (input_shape, input_shape)

        return images, bboxes, y_trues


if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    from utils import get_anchors
    from PyUtils.logs.print import *
    from PyUtils.viz.plot_draw import heatmap
    
    
    train_annotation_path = '../../data/2007_train.txt'
    with open(train_annotation_path, encoding='utf-8') as f:
        lines = f.readlines()

    input_shape     = [640, 640]
    num_classes = 20
    anchors_path    = '../../model_data/yolo_anchors.txt'
    anchors, num_anchors     = get_anchors(anchors_path)
    UnFreeze_Epoch      = 300
    batch_size = 2
    num_workers = 1
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7

    test_dataset   = YoloDataset(
        lines, 
        input_shape, 
        num_classes, 
        anchors, 
        anchors_mask, 
        # epoch_length=UnFreeze_Epoch,
        mosaic=mosaic, 
        mixup=mixup, 
        mosaic_prob=mosaic_prob, 
        mixup_prob=mixup_prob, 
        train=True, 
        # special_aug_ratio=special_aug_ratio
        fixed_input_shape=True
    )
    loader = DataLoader(
        test_dataset, 
        shuffle = shuffle, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        pin_memory=True,
        drop_last=True, 
        collate_fn=test_dataset.yolo_dataset_collate
    )
    for idx, batch in enumerate(loader):
        sllog << f'[{idx}]  {type(batch)}  {batch[0].shape=}'
        sllog << f'[{idx}]  {len(batch[1])}  {batch[1][0].shape}'
        
        break
