# ##################################################
#
# ##################################################
#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np


def grid_map(data, shape):
    
    norm_img = np.zeros(data.shape)
    cv2.normalize(data , norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像

    cv2.imwrite(f'/data/ylw/code/pl_yolo_v5/test/grid_map-1.jpg', heat_img)

    grid_delta = int(512/h)
    org_delta = int(512 / h / 2)
    heat_img = cv2.resize(heat_img, shape, interpolation=cv2.INTER_AREA)
    for i in range(h):
        for j in range(w):
            heat_img = cv2.putText(
                heat_img, 
                f'{int(data[i, j])}', 
                # (int(j*grid_delta+0), int(i*grid_delta+org_delta)),
                (j*grid_delta+0, i*grid_delta+org_delta),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.8,
                color=(255, 255, 255)
            )
    
    return heat_img


if __name__ == '__main__':
    
    data = np.linspace(0, 99, 100).reshape((10, 10))
    
    img = grid_map(data, shape=(512, 512))
    
    cv2.imwrite(f'/data/ylw/code/pl_yolo_v5/test/grid_map.jpg', img)
    # cv2.imshow('grid map', img)
    # if cv2.waitKey() == ord('q'):
    #     sys.exit(0)