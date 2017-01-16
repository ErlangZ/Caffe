#!/bin/bash
set -x 
../../build/install/bin/caffe test --model train_val.prototxt -weights ./cifar10_full_iter_21911.caffemodel.h5  -iterations 10
