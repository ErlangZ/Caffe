#!/bin/bash
set -x 
../../build/install/bin/caffe test --model test_val.prototxt -weights ../../examples/cifar10/cifar10_full_iter_10000.caffemodel  -iterations 100
