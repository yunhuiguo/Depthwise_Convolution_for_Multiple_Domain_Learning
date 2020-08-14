import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import sys
sys.path.append('../')
import config_task
import math
import torch.nn.functional as F

from models.base import DownsampleB, conv3x3, BasicBlock, SeparableConv2d

class FlatResNet(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, task_id, dataset=None, dropout_list=None):
	x = self.seed(x, task_id)
        
	for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                x = F.relu(residual + self.blocks[segment][b](x, task_id, dataset, dropout_list))
		
        x = self.bns[task_id](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
	if dataset is not None:
		if dataset in dropout_list: 
			x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fcs[task_id](x)
        return x

class FlatResNet26(FlatResNet):
    def __init__(self, block, layers, num_classes = [10]):
        super(FlatResNet26, self).__init__()

        self.nb_tasks = len(num_classes)
	
        factor = config_task.factor
        self.in_planes = int(64*factor)
        self.conv1 = SeparableConv2d(3, int(64*factor), 3, 1, 1, 1, False, self.nb_tasks)
        self.pre_bn = nn.ModuleList([nn.BatchNorm2d(int(64*factor)) for i in range(self.nb_tasks)])
        strides = [2, 2, 2]
        filt_sizes = [128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(512*factor)), nn.ReLU(True)) for i in range(self.nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fcs = nn.ModuleList([nn.Linear(int(512*factor), num_classes[i]) for i in range(self.nb_tasks)])         

        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x, task_id):
        x = self.pre_bn[task_id](self.conv1(x, task_id))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes:
            downsample = DownsampleB(self.in_planes, planes, 2)

        layers = [block(self.in_planes, planes, self.nb_tasks, stride)]
        self.in_planes = planes 
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.nb_tasks))

        return layers, downsample
