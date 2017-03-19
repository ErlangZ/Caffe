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
import os
import caffe
import numpy as np
net_def_prototxt = '/home/erlangz/ErlangZ/Code/Caffe/models/Yolo/yolo_deploy.prototxt'
trained_net_caffemodel = '/home/erlangz/ErlangZ/Code/Caffe/models/Yolo/yolo_iter_5000.caffemodel'
images_dir = '/home/erlangz/ErlangZ/Code/Caffe/models/Pascal/VOC2012/JPEGImages/'
MEAN_PROTO_PATH = 'imagenet.mean' 
MEAN_NPY_PATH = 'mean.npy'                         # 转换后的numpy格式图像均值文件路径

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
#mean_file = np.array([104,117,123]) 
transformer.set_mean('data', np.load(MEAN_NPY_PATH).mean(1).mean(1)) #### subtract mean ####
transformer.set_raw_scale('data', 255.0) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR
transformer.set_input_scale('data', 1.0/255) # pixel value range

# set test batchsize
batchsize = 1
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class Box(object):
    def __init__(self, buf, conf, x, y, w, h):
        self.buf = buf
        self.conf = conf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.get_class()
    
    def class_grid(self):
        x = int(self.x * 7)
        y = int(self.y * 7)
        if x == 7: x = 6
        if y == 7: y = 6 
        return x, y 

    def get_class(self):
        x, y = self.class_grid()
        index = (x * 7 + y) * (2 * 5 + 20) + 10
        self.poss = -1.0
        self.t = -1
        for i in xrange(20):
            if self.buf[0, index + i] > self.poss:
                self.poss = self.buf[0, index + i]
                self.t = i

    def corectness(self):
        return self.poss * self.conf

    def draw(self, image):
        width = image.shape[1]
        height = image.shape[0]
        print "XXXXXXX", width, " ", height, " ", self.x, " ", self.y, " ", self.w, " ", self.h
        x_min = int((self.x - self.w / 2.0) * width)
        y_min = int((self.y - self.h / 2.0) * height)
        x_max = int((self.x + self.w / 2.0) * width)
        y_max = int((self.y + self.h / 2.0) * height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
        t = Types[self.t] + ":" + str(self.conf * self.poss)
        cv2.putText(image, t, (x_min + 4, y_min + 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


def confidence(buf):
    print buf[0:50]
    for i in xrange(7):
        for j in xrange(7):
            offset = (i * 7 + j) * (2 * 5 + 20)
            yield Box(buf, buf[0, offset], buf[0, offset+1], buf[0, offset+2], buf[0, offset+3],buf[0, offset+4])
            yield Box(buf, buf[0, offset + 5], buf[0, offset+6], buf[0, offset+7], buf[0, offset+8],buf[0, offset+9])

if __name__ == "__main__":
    # load data, len(images) = batchsize
    for f in os.listdir(images_dir):
        images = [os.path.join(images_dir, f),]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        #print net.blobs['data'].data
 
        # process the data through network
        out = net.forward()

        im = cv2.imread(images[0]) 
        print images[0], im.shape
        for box in confidence(net.blobs['ip2'].data):
            if box.corectness() > 0.2:
                box.draw(im)
        cv2.imshow("image", im)
        cv2.waitKey(0)
            #print box
        #print out 

        #result = [(sigmoid(float(out[k])), out[k], Types[int(k.split("-")[1])-1], ) for k in out]
        #result = [(sigmoid(float(out[k])), Types[int(k.split("-")[1])-1], ) for k in out]
        #result = sorted(result, reverse=True)    
        #print [i for i in result if i[0] >= 0.1]

