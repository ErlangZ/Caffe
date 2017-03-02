#!/bin/bash
set -x
export GLOG_log_dir=log
../../build/install/bin/caffe train  --solver=solver.prototxt  --snapshot=./caffe_yolo_train_iter_192.solverstate $@
#../../build/install/bin/caffe train  --solver=solver.prototxt   $@
