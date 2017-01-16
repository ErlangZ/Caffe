#!/bin/bash
set -x
../../build/install/bin/caffe train  --solver=solver.prototxt $@ > log/caffe.log 2>&1 
