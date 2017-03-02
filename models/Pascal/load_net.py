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
Names = {"aeroplane": 0, 
        "person": 1,
        "bicycle": 2, 
        "boat": 3,
        "bird": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19 }
Types = dict([(Names[name], name) for name in Names])
import sys
sys.path.append('/home/erlangz/Caffe/build/install/python/')
net_def_prototxt = '/home/erlangz/Caffe/models/Pascal/yolo_deploy.prototxt'
trained_net_caffemodel = '/home/erlangz/Caffe/models/Pascal/caffe_yolo_train_iter_152.caffemodel'
images_dir = '/home/erlangz/darknet/Data/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/'

import caffe
import numpy as npy
import os

import cv2

# setup net with ( structure definition file ) + ( caffemodel ), in test mode
net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)
# add preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
#mean_file = np.array([104,117,123]) 
#transformer.set_mean('data', mean_file) #### subtract mean ####
transformer.set_raw_scale('data', 255.0) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

# set test batchsize
batchsize = 1
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

if __name__ == "__main__":
    # load data, len(images) = batchsize
    for f in os.listdir(images_dir):
        images = [os.path.join(images_dir, f),]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        #print net.blobs['data'].data
 
        # process the data through network
        out = net.forward()
        print f, 
        result = [(sigmoid(float(out[k])), out[k], Types[int(k.split("-")[1])-1], ) for k in out]
        result = sorted(result, reverse=True)    
        print [i for i in result if i[0] >= 0.5]

