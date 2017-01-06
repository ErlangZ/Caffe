#!/bin/bash
set -x
../../build/install/bin/caffe train  --solver=solver.prototxt $@
