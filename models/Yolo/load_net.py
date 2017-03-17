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
net_def_prototxt = '/home/erlangz/Caffe/models/Yolo/yolo_tiny.prototxt'
trained_net_caffemodel = '/home/erlangz/Caffe/models/Yolo/caffe_yolo_train_iter_220.caffemodel'
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
transformer.set_raw_scale('data', 1.0) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

# set test batchsize
batchsize = 1
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class Box(object):
    def __init__(self, buf, x, y, w, h):
        self.buf = buf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def class_grid(self):
        x = int(self.x * 7)
        y = int(self.y * 7)
        if x == 7: x = 6
        if y == 7: y = 6 
        return x, y 

    def get_class(self):
        x, y = self.class_grid()
        index = (x * 7 + y) * (2 * 5 + 20) + 10
        poss = -1.0
        t = -1
        for i in xrange(20):
            if self.buf[0, index + i] > poss:
                poss = self.buf[0, index + i]
                t = i
        return t

    def draw(self, image):
        width = image.shape[0]
        height = image.shape[1]
        x_min = int((self.x - self.w / 2.0) * width)
        y_min = int((self.y - self.h / 2.0) * height)
        x_max = int((self.x + self.w / 2.0) * width)
        y_max = int((self.y + self.h / 2.0) * height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255))
        t = Types[self.get_class()]
        cv2.putText(image, t, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def confidence(buf, threshold):
    for i in xrange(7):
        for j in xrange(7):
            offset = i * 7 + j
            if (buf[0, offset] > threshold):
                yield Box(buf, buf[0, offset+1], buf[0, offset+2], buf[0, offset+3],buf[0, offset+4])
            if (buf[0, offset + 5] > threshold):
                yield Box(buf, buf[0, offset+6], buf[0, offset+7], buf[0, offset+8],buf[0, offset+9])

if __name__ == "__main__":
    # load data, len(images) = batchsize
    for f in os.listdir(images_dir):
        images = [os.path.join(images_dir, f),]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        #print net.blobs['data'].data
 
        # process the data through network
        out = net.forward()

        im = cv2.imread(images[0]) 
        print dir(im)
        print im.shape
        for box in confidence(net.blobs['ip2'].data, 0.6):
            box.draw(im)
        cv2.imshow("image", im)
        cv2.waitKey(0)
            #print box
        #print out 

        #result = [(sigmoid(float(out[k])), out[k], Types[int(k.split("-")[1])-1], ) for k in out]
        #result = [(sigmoid(float(out[k])), Types[int(k.split("-")[1])-1], ) for k in out]
        #result = sorted(result, reverse=True)    
        #print [i for i in result if i[0] >= 0.1]

