#!/usr/bin/env sh

GLOG_logtostderr=1 /home/me/caffe/build/tools/caffe train \
    --solver=solver.prototxt \
   --weights=bvlc_googlenet.caffemodel --gpu=0
