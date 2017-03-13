// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/03/13 13:38:23
// @file src/caffe/test/test_yolo_data_layer.cpp
// @brief 
// 
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/yolo_data_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class YoloDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  YoloDataLayerTest<TypeParam>() {

  }
  virtual ~YoloDataLayerTest<TypeParam>() {

  }
};

TYPED_TEST_CASE(YoloDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloDataLayerTest, TestSetup) {

}

TYPED_TEST(YoloDataLayerTest, TestReshape) {
   typedef typename TypeParam::Dtype Dtype;
   LayerParameter layer_param;
   layer_param.mutable_data_param()->set_batch_size(16); 
   std::string db_file(CMAKE_SOURCE_DIR "/caffe/test/test_data/yolo_train_lmdb");

   layer_param.mutable_data_param()->set_source(db_file);
   layer_param.mutable_data_param()->set_backend(DataParameter::LMDB);
   layer_param.mutable_yolo_data_param()->set_max_labels_number(2); //2 * 5 + 1
   YoloDataLayer<Dtype> yolo_data_layer(layer_param);
   
   vector<Blob<Dtype>*> bottom;
   vector<Blob<Dtype>*> top(2, NULL);
   top[0] = new Blob<Dtype>();
   top[1] = new Blob<Dtype>();

   yolo_data_layer.LayerSetUp(bottom, top);
   for (int i = 0; i < BasePrefetchingDataLayer<Dtype>::PREFETCH_COUNT; i++) {
       EXPECT_EQ(16, yolo_data_layer.prefetch_[i].data_.num());
       EXPECT_EQ(3, yolo_data_layer.prefetch_[i].data_.channels());
       EXPECT_EQ(448, yolo_data_layer.prefetch_[i].data_.height());
       EXPECT_EQ(448, yolo_data_layer.prefetch_[i].data_.width());

       EXPECT_EQ(16, yolo_data_layer.prefetch_[i].label_.num());
       EXPECT_EQ(11, yolo_data_layer.prefetch_[i].label_.channels());
       EXPECT_EQ(1, yolo_data_layer.prefetch_[i].label_.height());
       EXPECT_EQ(1, yolo_data_layer.prefetch_[i].label_.width());
   }
   delete top[0];
   delete top[1];
}

}
