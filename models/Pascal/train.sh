#!/bin/bash
set -x
export GLOG_log_dir=log
../../build/install/bin/caffe train  --solver=solver.prototxt $@
