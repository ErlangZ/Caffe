#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

#DATA=/home/erlangz/darknet/Data/VOCdevkit/VOCdevkit/VOC2012/JPEGImages
DATA=`pwd`
TOOLS=/home/erlangz/Caffe/build/tools

TRAIN_DATA_ROOT=./train/
VAL_DATA_ROOT=./test/

# Set RESIZE=true to resize the images to 448*448. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=448
  RESIZE_WIDTH=448
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_pascal \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --max_count=2 \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $DATA/pascal_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_pascal \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --max_count=2 \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $DATA/pascal_val_lmdb

echo "Done."
