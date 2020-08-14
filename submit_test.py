import torch 
from torch.autograd import Variable                                                                                            
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import numpy as np
import json
import collections

import torch.utils.data as data
import pickle
import config_task
from PIL import Image
from pycocotools.coco import COCO
import os.path

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Residual Adapters training')
parser.add_argument('--datadir', default='../ibm_project/residual_adapters/data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='../ibm_project/residual_adapters/data/decathlon-1.0/annotations/', help='annotation folder')

parser.add_argument('--seed', default=0, type=int, help='seed')
args = parser.parse_args()

datasets = [
    ("aircraft", 0),
    ("cifar100", 1),
    ("daimlerpedcls", 2),
    ("dtd", 3),
    ("gtsrb", 4),
    ("imagenet12", 5),
    ("omniglot", 6),
    ("svhn", 7),
    ("ucf101", 8),
    ("vgg-flowers", 9)]

num_classes = [100, 100, 2, 47, 43, 1000, 1623, 10, 101, 102]

datasets = collections.OrderedDict(datasets)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, index=None,
            labels=None ,imgs=None,loader=pil_loader,skip_label_indexing=0):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        if index is not None:
            imgs = [imgs[i] for i in index]
        self.imgs = imgs

        if index is not None:
            if skip_label_indexing == 0:
                labels = [labels[i] for i in index]

       #self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index][0]
        #target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        #return img, target, self.imgs[index][1]
        return img, self.imgs[index][1]

    def __len__(self):
        return len(self.imgs)

def prepare_data_loaders(dataset_names, data_dir, imdb_dir, shuffle_train=True, index=None):
    val_loaders = []
    #num_classes = []
    val = [0]
    config_task.offset = []

    imdb_names_val   = [imdb_dir + '/' + dataset_names[i] + '_test_stripped.json' for i in range(len(dataset_names))]
    imdb_names = [imdb_names_val]

    with open(data_dir + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle.load(handle)
    
    for i in range(len(dataset_names)):
        imgnames_val = []
        for itera1 in val:
            annFile = imdb_names[itera1][i]
            coco = COCO(annFile)
            imgIds = coco.getImgIds()
            annIds = coco.getAnnIds(imgIds=imgIds)
            anno = coco.loadAnns(annIds)
            images = coco.loadImgs(imgIds) 
            timgnames = [img['file_name'] for img in images]
            timgnames_id = [img['id'] for img in images]
	    #labels = [int(ann['category_id'])-1 for ann in anno]
            #min_lab = min(labels)
            #labels = [lab - min_lab for lab in labels]
            #max_lab = max(labels)
            imgnames = []
            for j in range(len(timgnames)):
                imgnames.append((data_dir + '/' + timgnames[j],timgnames_id[j]))

            if itera1 in val:
                imgnames_val += imgnames
        means = dict_mean_std[dataset_names[i] + 'mean']
        stds = dict_mean_std[dataset_names[i] + 'std']

        #means = [0.485, 0.456, 0.406]
        #stds = [0.229, 0.224, 0.225]

        if dataset_names[i] in ['gtsrb', 'omniglot','svhn']: # no horz flip 
            transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])
        elif dataset_names[i] in ['daimlerpedcls']: # no horz flip 
            transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(72),
            	transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        
        img_path = data_dir
        valloader = torch.utils.data.DataLoader(ImageFolder(data_dir, transform_test, None, None, None, imgnames_val), batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        val_loaders.append(valloader) 
    
    return val_loaders #, num_classes

def test(val_loader, net, task_id):
    net.eval()
    # Test the model
    with torch.no_grad():
        losses = AverageMeter() 
        top1 = AverageMeter()
        for idx, (images, image_ids) in enumerate(val_loader):
            if use_cuda:
                images  = images.cuda(async=True)
                images  = Variable(images, volatile=True)

	        outputs = net.forward(images, task_id=0)
            _, predicted = torch.max(outputs.data, 1)
	    
            for image_idx, prediction in enumerate(predicted.data.cpu().numpy()):
                res_dict = {}
                res_dict['category_id'] = int(10e6 * (task_id+1) + prediction + 1)
                res_dict['image_id'] = image_ids.data.cpu().numpy()[image_idx]
                results.append(res_dict)

	    #correct = predicted.eq(labels.data).cpu().sum()
            #top1.update(correct.item()*100/(labels.size(0)+0.0), labels.size(0))

            # Loss
            #loss = criterion(outputs, labels)
            #losses.update(loss.data[0], labels.size(0))
        
        #print ("Loss: {:.4f}, Acc Avg: {:.4f}"
        #    .format(losses.val, top1.avg))

#####################################
# Prepare data loaders
#val_loaders, num_classes = prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)
val_loaders = prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)

results = []
for dataset in datasets.keys():
    print dataset 
    #num_class = [num_classes[datasets[dataset]]]
    num_class = [num_classes[datasets[dataset]]]
    task_id = datasets[dataset]
    # Create the network
    net = get_model('Resnet26', num_class, dataset, mode='submit')

    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        #net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    #validation
    test(val_loaders[datasets[dataset]], net, task_id) 
    
f =  "./results.json"
with open(f, 'wb') as fh:
    json.dump(results, fh)     
