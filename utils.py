import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
import shutil

from models.base import DownsampleB, conv3x3, BasicBlock
from models import resnet
from itertools import cycle

from models import resnet_sharing
from models.base_sharing import BasicBlock_sharing

def MinibatchScheduler(tloaders, mode = 'cycle'):
    if len(tloaders) == 1:
    	for i, data_pair in enumerate(tloaders[0]):
             yield i, [data_pair]
    else:	
    	if mode == 'cycle':
            s = max((len(l), i) for i, l in enumerate(tloaders))[1]
            tmp = tloaders[0]
            tloaders[0] = tloaders[s]
            tloaders[s] = tmp
            data = []

            ziplist = zip(tloaders[0], cycle(tloaders[1]))
            for i in range(2, len(tloaders)):
                flatlist = zip(*ziplist)
                flatlist.append(cycle(tloaders[i]))
                ziplist = zip(*flatlist)

            for i, data_pair in enumerate(ziplist):
                 yield i, data_pair

        elif mode == 'min':
            for i, data_pair in enumerate(zip(*tloaders)):
                yield i, data_pair

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, sparsity, variance, policy_set

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print '# setting learning_rate to %.2E'%lr

def adjust_learning_rate_and_learning_taks(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    '''
    if epoch >= args.step3:
        lr = args.lr * 0.001
    if epoch >= args.step2:
        lr = args.lr * 0.01        
    '''
    if epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(source, net, num_classes, dataset, mode):
    checkpoint = torch.load(source)
    net_old = checkpoint['net']
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1
    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
            	m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    if mode == 'submit':
    	#if dataset is 'imagenet12' or dataset is 'daimlerpedcls':
    	net.fcs[0].weight.data = torch.nn.Parameter(net_old.fcs[0].weight.data)
    	net.fcs[0].bias.data = torch.nn.Parameter(net_old.fcs[0].bias.data)  
	'''
    	else:
    
    		net.fcs[0].weight.data = torch.nn.Parameter(net_old.module.fcs[0].weight.data)
    		net.fcs[0].bias.data = torch.nn.Parameter(net_old.module.fcs[0].bias.data)  
	'''
    del net_old
    return net

def submit_sharing(source, net, num_classes, dataset, mode):
    checkpoint = torch.load(source)
    net_old = checkpoint['net']

    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1
    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d): # and 'bn' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if mode == 'submit':
        store_weight = []
        store_bias = []
        for name, m in net_old.named_modules():
            if isinstance(m, nn.Linear): 
                store_weight.append(m.weight.data)
                store_bias.append(m.bias.data)

        element = 0
        for name, m in net.named_modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.Parameter(store_weight[element])
                m.bias.data = torch.nn.Parameter(store_bias[element])  
                element += 1

    del net_old
    return net

def get_model(model, num_classes, dataset = None, mode = 'train', dropout_list=None):
    if model == 'Resnet26':
        layer_config = [4, 4, 4]
	#if dataset == 'imagenet12':
        rnet = resnet.FlatResNet26(BasicBlock, layer_config, num_classes)
	#else:
        #rnet = resnet_sharing.FlatResNet26(BasicBlock_sharing, layer_config, num_classes)

        if dataset is not None:
            if dataset == 'imagenet12':
            	source = './pretrained_model_on_train_and_val/imagenet12_separableConv/' +  dataset + '.t7'
	    else:
            	source = './depthwise_sharing/' + dataset + '/' + dataset + '.t7'
        # load pretrained weights into flat ResNet
	#if dataset == "imagenet12":
        rnet = load_weights_to_flatresnet(source, rnet, num_classes, dataset, mode)
	#else:
        #rnet = submit_sharing(source, rnet, num_classes, dataset, mode)
    return rnet

def load_sharing(dataset, net, num_classes, model_name):

    source = './log/' + dataset + '/' + dataset + '.t7'
    checkpoint = torch.load(source)
    net_old = checkpoint['net']

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d): 
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    
    store_data_point = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1):
            store_data_point.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==1):
            m.weight.data = torch.nn.Parameter(store_data_point[element])
            element += 1
   
    net.fcs[0].weight.data = torch.nn.Parameter(net_old.fcs[0].weight.data)
    net.fcs[0].bias.data = torch.nn.Parameter(net_old.fcs[0].bias.data)  
    del net_old
	
    store_data_depth = []
    for dataset_ in model_name: 
	depth = []
	if dataset_ == "imagenet12":
    		source = './pretrained_model_on_train_and_val/imagenet12_separableConv/' +  dataset_ + '.t7'	
	else:
		source = './log/' + dataset_ + '/' + dataset_ + '.t7'
    	checkpoint = torch.load(source)
    	net_old = checkpoint['net']
		
    	for name, m in net_old.named_modules():
        	if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            		depth.append(m.weight.data)
	store_data_depth.append(depth)
    	del net_old

    element = 0
    count = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.data = torch.nn.Parameter(store_data_depth[count][element])
	    count += 1
	    if count == 10:
		element += 1
		count = 0
    return net


def get_sharing_model(model, num_classes, dataset = None, mode = 'train', dropout_list=None):
    
    model_name = ["aircraft", "cifar100", "daimlerpedcls", "dtd", "gtsrb", "omniglot", \
        "svhn", "ucf101",  "vgg-flowers", "imagenet12"]
        
    if model_name[0] != dataset:
    	idx = model_name.index(dataset)
    	temp = model_name[0]
    	model_name[0] = model_name[idx]
    	model_name[idx] = temp

    if model == 'Resnet26':
        layer_config = [4, 4, 4]
        rnet = resnet_sharing.FlatResNet26(BasicBlock_sharing, layer_config, num_classes)
        rnet = load_sharing(dataset, rnet, num_classes, model_name)
    return rnet
