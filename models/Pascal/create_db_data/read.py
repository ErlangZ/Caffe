#!/usr/bin/env python
# -*- coding: utf8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: read.py
Author: erlangz(erlangz@baidu.com)
Date: 2017/02/28 16:30:18
"""
import lmdb
import caffe
import cv2
import numpy as np

env = lmdb.open('/home/erlangz/Caffe/models/Pascal/create_db_data/pascal_train_lmdb', readonly=True)
with env.begin() as contxt:
    for k, v in contxt.cursor():
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(v)
        print "read an image:"
        print len(datum.multi_labels)
        print len(datum.data)
        print "XXXXXXXXXXXX"
        #data = caffe.io.datum_to_array(datum)
        #im = data.astype(np.uint8)
        #np.transpose(im, (2, 1, 0))
        #cv2.imshow("windows", im) 
        #cv2.waitKey(0)
        #cv2.destroyWindow("windows")
