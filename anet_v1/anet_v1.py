import os,sys
import torch
import numpy as np
import cv2





class Block(torch.nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=0, bias=False, bn=False, act='relu') -> None:
        super(Block, self).__init__()
        self.ic = ic
        self.oc = oc
        self.bn = bn
        self.c = torch.nn.Conv2d(self.ic, self.oc, kernel_size=k, stride=s, padding=p, bias=bias)
        if bn:
            self.b = torch.nn.BatchNorm2d(self.oc)
        self.act = None
        if act == 'relu':
            self.act = torch.nn.ReLU()

    def forward(self, x):
        y = self.c(x)
        if self.bn:
            y = self.b(y)
        if self.act:
            y = self.act(y)
        return y


class ABlock(torch.nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=0, bias=False, bn=False, ali=0, alo=0, keep=False, act='relu') -> None:
        super(ABlock, self).__init__()
        self.ic = ic
        self.oc = oc
        self.bn = bn
        # p = 0
        # if keep:
        #     p = 1  # TODO
        self.c = torch.nn.Conv2d(self.ic, self.oc, kernel_size=k, stride=s, padding=p, bias=bias)
        self.ac = torch.nn.Conv2d(self.ic, self.oc, kernel_size=5, stride=5, padding=0, bias=bias)
        self.al = torch.nn.Linear(in_features=ali, out_features=alo, bias=False)
        if bn:
            self.b = torch.nn.BatchNorm2d(self.oc)
        if act == 'relu':
            self.act = torch.nn.ReLU()

    def forward(self, x):
        y = self.c(x)
        a = self.ac(x)
        # print(f'[y]  {y.shape=}')
        # print(f'{y.shape=}  {a.shape=}')
        a = a.reshape(a.shape[0], a.shape[1], -1)
        # print(f'[reshape-a]  {a.shape=}')
        a = self.al(a)
        # print(f'[al]  {a.shape=}')
        y *= a.view(a.shape[0], a.shape[1], 1, 1)
        # print(f'[y]  {y.shape=}')
        if self.bn:
            y = self.b(y)
        # print(f'[y]  {y.shape=}')
        if self.act:
            y = self.act(y)
        return y



class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.c1 = Block(3, 64, 3, 2, 0, bias=False, bn=True)
        
        self.c2 = Block(64, 128, 3, 1, 1, bias=False, bn=True)
        self.c3 = Block(128, 128, 3, 1, 1, bias=False, bn=True)
        self.c4 = Block(128, 128, 3, 1, 1, bias=False, bn=True)
        
        self.c5= Block(128, 256, 3, 2, 0, bias=False, bn=True)
        self.c6= Block(256, 256, 3, 1, 1, bias=False, bn=True)
        self.c7= Block(256, 256, 3, 1, 1, bias=False, bn=True)
        
        self.c8= Block(256, 512, 3, 2, 0, bias=False, bn=True)
        self.c9= Block(512, 512, 3, 1, 1, bias=False, bn=True)
        self.c10= Block(512, 512, 3, 1, 1, bias=False, bn=True)
        
        self.c11= Block(512, 1024, 3, 2, 0, bias=False, bn=True)
        self.c12= Block(1024, 1024, 3, 1, 1, bias=False, bn=True)
        self.c13= Block(1024, 1024, 3, 1, 1, bias=False, bn=True)
        
        self.c14= Block(1024, 1024, 3, 1, 1, bias=False, bn=True)
        self.c15= Block(1024, 256, 3, 1, 1, bias=False, bn=True)
        self.c16= Block(256, 128, 3, 1, 1, bias=False, bn=True)
        self.feat = torch.nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        y1 = self.c1(x)
        print(f'{y1.shape=}')

        y2 = self.c2(y1)
        print(f'{y2.shape=}')
        y3 = self.c3(y2)
        print(f'{y3.shape=}')
        y4 = self.c4(y3)
        print(f'{y4.shape=}')
        y34 = y3+y4
        print(f'{y34.shape=}')

        y5 = self.c5(y34)
        print(f'{y5.shape=}')
        y6 = self.c6(y5)
        print(f'{y6.shape=}')
        y7 = self.c7(y6)
        print(f'{y7.shape=}')
        y67 = y6+y7
        print(f'{y67.shape=}')

        y8 = self.c8(y67)
        print(f'{y8.shape=}')
        y9 = self.c9(y8)
        print(f'{y9.shape=}')
        y10 = self.c10(y9)
        print(f'{y10.shape=}')
        y910 = y9+y10
        print(f'{y910.shape=}')

        y11 = self.c11(y910)
        print(f'{y11.shape=}')
        y12 = self.c12(y11)
        print(f'{y12.shape=}')
        y13 = self.c13(y12)
        print(f'{y13.shape=}')
        y1213 = y12+y13
        print(f'{y1213.shape=}')

        y14 = self.c14(y1213)
        y15 = self.c15(y14)
        y16 = self.c16(y15)
        y = self.feat(y16)
        # print(f'{y.shape=}')
        # y = y.view(y.shape[0], -1)
        print(f'{y.shape=}')
        
        return y

if __name__ == '__main__':
    
    x = torch.randn(size=(2, 3, 128, 128))
    print(f'{x.shape=}')
    # m = ABlock(3, 6, 3, 1, bias=False, bn=True, ali=36, alo=1, keep=True)
    
    # y = m(x)
    # print(f'{y.shape=}')

    m = Net()
    y = m(x)
    print(f'{y.shape=}')
    
