# File allowing to change which task is currently used for training/testing
import numpy as np

#aircraft  cifar100  daimlerpedcls  dtd	gtsrb  imagenet12  omniglot  svhn  ucf101  vgg-flowers

task = 0
mode = 'normal'
factor = 1.
wd3x3 = 1.0
decay3x3 = np.array(wd3x3) * 0.0001

# target rate
t = 0.6
