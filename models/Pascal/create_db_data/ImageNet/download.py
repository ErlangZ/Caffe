#!/usr/bin/env python
# -*- coding: utf8 -*-
########################################################################
# 
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: download.py
Author: erlangz(erlangz@baidu.com)
Date: 2017/03/03 16:18:55
"""
import os
import sys
import requests

def download_file(url, dst, params={}, debug=True):
    if debug:
        print u"downloading {0}...".format(dst),
    response = requests.get(url, params=params, timeout=1)
    content_type = response.headers["content-type"]
    if content_type.startswith("text"):
        raise TypeError("404 Error")
    with file(dst, "wb") as fp:
        fp.write(response.content)

def download_dir(filename, dirname):
    i = 1
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(filename) as f:
        for line in f.xreadlines():
            line = line.strip()
            print line
            if not (line.endswith(".jpg") or line.endswith(".JPG")):
                print >> sys.stderr, ("Skip Download:", line)
            else:
                sav_name = os.path.join(dirname, str(i)+".jpg")
                try:
                    download_file(line, sav_name) 
                    i += 1
                except:
                    print >> sys.stderr, ("Download:", line, "error")
                    
    return i

def download(dir):
    return download_dir(os.path.join(dir, "lists.txt"), os.path.join(dir, "data"))
from multiprocessing.pool import Pool
dirs = [f for f in os.listdir(".") if os.path.isdir(f)]
pool = Pool(19)
print pool.map(download, dirs)
pool.close()
pool.join()
