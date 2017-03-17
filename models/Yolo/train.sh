#!/bin/bash
set -x
export GLOG_log_dir=log
../../build/install/bin/caffe train  --solver=solver.prototxt  --snapshot=./caffe_yolo_train_iter_358.solverstate $@
#../../build/install/bin/caffe train  --solver=solver.prototxt  --weights=./caffe_yolo_train_iter_12323.caffemodel $@
#../../build/install/bin/caffe train  --solver=solver.prototxt   $@
