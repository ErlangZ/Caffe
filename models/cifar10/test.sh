#!/bin/bash
set -x 
../../build/install/bin/caffe test --model test_val.prototxt -weights caffe_alexnet_train_iter_100000.caffemodel  -iterations 100
