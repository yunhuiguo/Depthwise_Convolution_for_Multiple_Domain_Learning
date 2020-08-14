import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.utils as torchutils
from torch.nn import init, Parameter

import sys
sys.path.append('../')
import config_task

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0),1)

def depthwise_conv(in_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels, in_channels, kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)

def pointwise_conv(in_channels,out_channels,bias):
    return nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1,stride=1,padding=0,dilation=1,bias=False, nb_tasks=10):
        super(SeparableConv2d,self).__init__()
		
        self.relu = nn.ReLU(inplace=True)

        self.depthwise = nn.ModuleList([depthwise_conv(in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias) for i in range(nb_tasks)])  
        self.pointwise = pointwise_conv(in_channels, out_channels, bias)
    
    def forward(self, x, task_id):
        x = self.depthwise[task_id](x)
        x = self.pointwise(x)
		
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_filters, out_filters, nb_tasks=10, stride=1):
        super(BasicBlock, self).__init__()

        self.SeparableConv2d1 = SeparableConv2d(in_filters,out_filters, 3, stride=stride, padding=1, bias=False, nb_tasks=nb_tasks)
        self.bns1 = nn.ModuleList([nn.BatchNorm2d(out_filters) for i in range(nb_tasks)])
        self.relu = nn.ReLU(inplace=True)
	self.SeparableConv2d2 = SeparableConv2d(out_filters,out_filters, 3, stride=1, padding=1, bias=False, nb_tasks=nb_tasks)
        self.bns2 = nn.ModuleList([nn.BatchNorm2d(out_filters) for i in range(nb_tasks)])
        
    def forward(self, x, task_id, dataset= None, dropout_list=None):

    	x = self.SeparableConv2d1(x, task_id)
    	x = self.bns1[task_id](x)
    	x = self.relu(x)
	'''	
        if dataset is not None:
            if dataset in dropout_list: 
                x = F.dropout(x, p = 0.5, training = self.training)
	'''
    	x = self.SeparableConv2d2(x, task_id)
    	out = self.bns2[task_id](x)
        
        return out
