#!/usr/bin/env python
# -*- coding: utf8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: load_net.py
Author: erlangz(erlangz@baidu.com)
Date: 2017/03/01 10:36:35
"""
import sys
sys.path.append('/home/erlangz/Caffe/build/install/python/')
net_def_prototxt = '/home/erlangz/Caffe/models/Pascal/yolo_deploy.prototxt'
trained_net_caffemodel = '/home/erlangz/Caffe/models/Pascal/caffe_yolo_train_iter_10000.caffemodel'
images_dir = '/home/erlangz/darknet/Data/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/'

batchsize = 1
import caffe
import numpy as npy
import os

import cv2
#mat = cv2.imread(os.path.join(images_dir, '2008_007788.jpg'))
#images = [ mat, ]
images = [os.path.join(images_dir, '2008_007788.jpg'),]

# setup net with ( structure definition file ) + ( caffemodel ), in test mode
net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)
# add preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
#mean_file = np.array([104,117,123]) 
#transformer.set_mean('data', mean_file) #### subtract mean ####
#transformer.set_raw_scale('data', 255) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

# set test batchsize
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

# load data, len(images) = batchsize
net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
# process the data through network
out = net.forward()
print out['labels']
for k in out:
    if 'ip2-' in k:
        print k, sigmoid(float(out[k]))

