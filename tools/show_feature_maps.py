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
sys.path.append('/home/erlangz/caffe/build/install/python/')
import cv2
import os
import caffe
import math
import numpy as np

net_def_prototxt = '/home/erlangz/caffe/build/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-deploy.prototxt'
trained_net_caffemodel = '/home/erlangz/caffe/build/deep-visualization-toolbox/models/caffenet-yos/caffenet-yos-weights'
MEAN_NPY_PATH = './ilsvrc_2012_mean.npy'        


net = caffe.Net(net_def_prototxt, trained_net_caffemodel, caffe.TEST)
# add preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
transformer.set_mean('data', np.load(MEAN_NPY_PATH).mean(1).mean(1)) #### subtract mean ####
transformer.set_raw_scale('data', 255.0) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR
transformer.set_input_scale('data', 1.0/255) # pixel value range

# set test batchsize
batchsize = 1
data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])



def normlize_pic(pic, scale_range=1.0):
    pic -= pic.min()
    pic *= scale_range / (pic.max() + 1e-10)
    return pic

def show_features(features, name, max_width=10):
    pic_numbers = features.shape[0]
    pic_h = features.shape[1]
    pic_w = features.shape[2]

    OutputPicCols = max_width 
    OutputPicRows = int(math.ceil(float(pic_numbers) / OutputPicCols))
    
    data = np.zeros((OutputPicRows * pic_h, OutputPicCols * pic_w))
    for pic_number in xrange(pic_numbers):
        pic = normlize_pic(features[pic_number])
        H = pic_number / OutputPicCols
        W = pic_number % OutputPicCols
        data[H*pic_h : (H+1)*pic_h, W*pic_w : (W+1)*pic_w] = pic
    cv2.imshow(name, data) 


if __name__ == "__main__":
    image_names = os.listdir('../../input_images/')
    #image_names = ['../../input_images/ILSVRC2012_val_00008338.jpg',]
    # load data, len(images) = batchsize
    for image in image_names:
        image = os.path.join('../../input_images', image)
        net.blobs['data'].data[...] = transformer.preprocess('data',caffe.io.load_image(image))
        # process the data through network
        out = net.forward()
        
        data = net.blobs['data'].data[0]
        show_features(data, "data", 3)

        conv1_feature_map = net.blobs['conv1'].data[0]  #batch_size=1
        show_features(conv1_feature_map, "conv1_feature")

        conv2_feature_map = net.blobs['conv2'].data[0]  #batch_size=1
        show_features(conv2_feature_map, "conv2_feature")

        conv3_feature_map = net.blobs['conv3'].data[0]  #batch_size=1
        show_features(conv3_feature_map, "conv3_feature")

        cv2.waitKey(0)

