#!/usr/bin/env sh
set -e

../../build/tools/caffe  test --model test.prototxt -weights ./lenet_iter_10000.caffemodel  -iterations 100
