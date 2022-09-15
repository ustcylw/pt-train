import os,sys
import torch
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F



from anet_v1 import Block, ABlock



class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.c1 = Block(1, 16, 3, 1, 1, bias=False, bn=True)
        
        self.c5= Block(16, 32, 3, 2, 0, bias=False, bn=True)
        self.c6= Block(32, 32, 3, 1, 1, bias=False, bn=True)
        self.c7= Block(32, 32, 3, 1, 1, bias=False, bn=True)
        
        self.c8= Block(32, 64, 3, 2, 0, bias=False, bn=True)
        self.c9= Block(64, 64, 3, 1, 1, bias=False, bn=True)
        self.c10= Block(64, 64, 3, 1, 1, bias=False, bn=True)

        self.c14= Block(64, 16, 3, 1, 1, bias=False, bn=True)
        self.feat = torch.nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        y1 = self.c1(x)
        # print(f'{y1.shape=}')

        y5 = self.c5(y1)
        # print(f'{y5.shape=}')
        y6 = self.c6(y5)
        # print(f'{y6.shape=}')
        y7 = self.c7(y6)
        # print(f'{y7.shape=}')
        y67 = y6+y7
        # print(f'{y67.shape=}')

        y8 = self.c8(y67)
        # print(f'{y8.shape=}')
        y9 = self.c9(y8)
        # print(f'{y9.shape=}')
        y10 = self.c10(y9)
        # print(f'{y10.shape=}')
        y910 = y9+y10
        # print(f'{y910.shape=}')

        y14 = self.c14(y910)
        y = self.feat(y14)
        # print(f'{y.shape=}')
        y = y.view(y.shape[0], -1)
        # print(f'{y.shape=}')
        
        return y



class ANet(torch.nn.Module):
    def __init__(self) -> None:
        super(ANet, self).__init__()
        self.c1 = ABlock(1, 16, 3, 1, 1, bias=False, bn=True, ali=25, alo=1)
        
        self.c5= ABlock(16, 32, 3, 2, 0, bias=False, bn=True, ali=25, alo=1)
        self.c6= ABlock(32, 32, 3, 1, 1, bias=False, bn=True, ali=4, alo=1)
        self.c7= ABlock(32, 32, 3, 1, 1, bias=False, bn=True, ali=4, alo=1)
        
        self.c8= ABlock(32, 64, 3, 2, 0, bias=False, bn=True, ali=4, alo=1)
        self.c9= ABlock(64, 64, 3, 1, 1, bias=False, bn=True, ali=1, alo=1)
        self.c10= Block(64, 64, 3, 1, 1, bias=False, bn=True)

        self.c14= Block(64, 16, 3, 1, 1, bias=False, bn=True)
        self.feat = torch.nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        y1 = self.c1(x)
        # print(f'='*80+'y1'+'*'*80+f'{y1.shape=}\n\n')

        y5 = self.c5(y1)
        # print(f'='*80+'y5'+'*'*80+f'{y5.shape=}\n\n')
        y6 = self.c6(y5)
        # print(f'='*80+'y6'+'*'*80+f'{y6.shape=}\n\n')
        y7 = self.c7(y6)
        # print(f'='*80+'y7'+'*'*80+f'{y7.shape=}\n\n')
        y67 = y6+y7
        # print(f'='*80+'y67'+'*'*80+f'{y67.shape=}\n\n')

        y8 = self.c8(y67)
        # print(f'='*80+'y8'+'*'*80+f'{y8.shape=}\n\n')
        y9 = self.c9(y8)
        # print(f'='*80+'y9'+'*'*80+f'{y9.shape=}\n\n')
        y10 = self.c10(y9)
        # print(f'='*80+'y10'+'*'*80+f'{y10.shape=}\n\n')
        y910 = y9+y10
        # print(f'='*80+'y910'+'*'*80+f'{y910.shape=}\n\n')

        y14 = self.c14(y910)
        y = self.feat(y14)
        # print(f'{y.shape=}')
        y = y.view(y.shape[0], -1)
        # print(f'='*80+'y'+'*'*80+f'{y.shape=}')
        
        return y


if __name__ == '__main__':
    
    x = torch.randn(size=(2, 1, 28, 28))
    print(f'{x.shape=}')

    m = ANet()
    y = m(x)
    print(f'{y.shape=}')
    
