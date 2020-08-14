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

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(FeedforwardNN, self).__init__()
        # reduction rate
        d = 4
        hidden_size = input_size / d + 1

        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

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
    return nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)

def pointwise_conv(in_channels,out_channels,bias):
    return nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1,stride=1,padding=0,dilation=1, bias=False, nb_tasks=10, cnt_depth=0):
        super(SeparableConv2d,self).__init__()
		
        self.relu = nn.ReLU(inplace=True)
        self.depthwise = nn.ModuleList([depthwise_conv(in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias) for i in range(nb_tasks)])  
        self.pointwise = pointwise_conv(in_channels, out_channels, bias)
        self.cnt_depth = cnt_depth

        self.action_network = FeedforwardNN(in_channels).cuda()
      	
	
    def forward(self, x, task_id):
	# cnt_depth: 0 - 24
        if self.cnt_depth >= 0 and self.cnt_depth <= 24:
            action_input = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  
            action_output = self.action_network(action_input)
            prob = F.softmax(action_output)

            results = []
            for ii, model in enumerate(self.depthwise):
                x_ = model(x)
            	v = prob[:,ii].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            	x_ = v * x_	    
            	results.append(x_)

            r = sum(results)
            x = self.pointwise(r)
        else:
            x = self.depthwise[task_id](x)
            x = self.pointwise(x)
        return x

class BasicBlock_sharing(nn.Module):

    def __init__(self, in_filters, out_filters, nb_tasks=10, stride=1, cnt_depth=0):
        super(BasicBlock_sharing, self).__init__()
        self.SeparableConv2d1 = SeparableConv2d(in_filters,out_filters, 3, stride=stride, padding=1, bias=False, nb_tasks=nb_tasks, cnt_depth=cnt_depth)
        cnt_depth += 1
        self.bns1 = nn.ModuleList([nn.BatchNorm2d(out_filters)])
        self.relu = nn.ReLU(inplace=True)
        self.SeparableConv2d2 = SeparableConv2d(out_filters,out_filters, 3, stride=1, padding=1, bias=False, nb_tasks=nb_tasks, cnt_depth=cnt_depth)
        self.bns2 = nn.ModuleList([nn.BatchNorm2d(out_filters)])
        
    def forward(self, x, task_id, dataset= None, dropout_list=None):

    	x = self.SeparableConv2d1(x, task_id)
    	x = self.bns1[task_id](x)
    	x = self.relu(x)
    	
	x = self.SeparableConv2d2(x, task_id)
    	out = self.bns2[task_id](x)
        
        return out
