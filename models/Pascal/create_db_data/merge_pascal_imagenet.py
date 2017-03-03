#!/usr/bin/env python
# -*- coding: utf8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: merge_pascal_imagenet.py
Author: erlangz(erlangz@baidu.com)
Date: 2017/03/03 17:56:02
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
index = [0 for i in xrange(20)]
import shutil
import os
with open("./labels.data") as f:
    for line in f.xreadlines():
        line = line.strip()
        data = line.split("\t")
        file_name = data[0].strip()
        labels = [int(i) for i in data[1:]]
        if sum(labels) == 1:
            label = [i for i, l in enumerate(labels) if l == 1][0]
            if label != 15: continue
            print file_name, "is single_labels:", Types[label]
            shutil.copyfile(file_name, os.path.join('/home/erlangz/Caffe/models/Pascal/create_db_data/ImageNet', Types[label], "pascal"+str(index[label])+".jpg"))
            index[label] += 1

