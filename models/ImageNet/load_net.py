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
sys.path.append('/home/erlangz/ErlangZ/Code/Caffe/build/install/python/')
net_def_prototxt = '/home/erlangz/ErlangZ/Code/Caffe/models/ImageNet/yolo_test.prototxt'
trained_net_caffemodel = '/home/erlangz/ErlangZ/Code/Caffe/models/ImageNet/caffe_yolo_train_iter_7300.caffemodel'
images_dir = '/home/erlangz/ErlangZ/Code/Caffe/models/Pascal/VOC2012/JPEGImages/'
MEAN_PROTO_PATH = '/home/erlangz/ErlangZ/Code/Caffe/models/ImageNet/mean.binaryproto'
MEAN_NPY_PATH = 'mean.npy'                         # 转换后的numpy格式图像均值文件路径

import numpy as np
import os

import caffe
blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
blob.ParseFromString(data)                         # 解析文件内容到blob

array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
np.save(MEAN_NPY_PATH, mean_npy)

import cv2

# setup net with ( structure definition file ) + ( caffemodel ), in test mode
net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)
# add preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
mean_npy = np.load(MEAN_NPY_PATH)
transformer.set_raw_scale('data', 255.0) # pixel value range
transformer.set_mean('data', mean_npy.mean(1).mean(1)) #### subtract mean ####
transformer.set_input_scale('data', 1.0/255.0) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

# set test batchsize
batchsize = 1
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def softmax(x):
    return math.exp(x)/sum(math.exp(x))

right_image = {}
all_image = {}
if __name__ == "__main__":
    import cv2
    # load data, len(images) = batchsize
    #for f in os.listdir(images_dir):
    images = []
    #with open("data/file_list.txt") as f:
    #with open("data/test_list.txt") as f:
    with open("data/train_list.txt") as f:
        for line in f.xreadlines():
            images.append(line.split("\t")[0])
    count = 0 
    for f in images:
        count += 1
        if count % 300 == 0:
            print count
        #images = [os.path.join(images_dir, f),]
        images = [os.path.join('data', f)]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        #print net.blobs['data'].data
        im = cv2.imread(images[0]) 
        # process the data through network
        out = net.forward()
        label = f.split("/")[0]
        if label not in right_image:
            right_image[label] = 0
            all_image[label] = 0
        if label == Types[int(out['output'].argmax())]:
            right_image[label] += 1
        all_image[label] += 1
        #cv2.imshow("output", im)
        #cv2.waitKey(0)
    for label in all_image:
        print label, float(right_image[label])/float(1+all_image[label])
