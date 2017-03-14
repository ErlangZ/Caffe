// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/03/14 14:10:39
// @file ../src/caffe/test/test_yolo_loss_layer.cpp
// @brief 
// 
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class YoloLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  YoloLossLayerTest<TypeParam>() {

  }
  virtual ~YoloLossLayerTest<TypeParam>() {

  }
};

TYPED_TEST_CASE(YoloLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloLossLayerTest, TestBoxInit) {
   typedef typename TypeParam::Dtype Dtype;

   Dtype data[5];
   data[0] = 12;
   data[1] = 0.2;
   data[2] = 0.3;
   data[3] = 0.4;
   data[4] = 0.5;
   YoloBox<Dtype> box1(data, true);
   EXPECT_EQ(12, box1.type);
   EXPECT_EQ(1.0, box1.confidence);
   EXPECT_NEAR(0.2, box1.center_x, 1e-6);
   EXPECT_NEAR(0.3, box1.center_y, 1e-6);
   EXPECT_NEAR(0.4, box1.width, 1e-6);
   EXPECT_NEAR(0.5, box1.height, 1e-6);

   data[0] = 0.1;
   YoloBox<Dtype> box2(data, false);
   EXPECT_EQ(-1, box2.type);
   EXPECT_NEAR(0.1, box2.confidence, 1e-6);
   EXPECT_NEAR(0.2, box2.center_x, 1e-6);
   EXPECT_NEAR(0.3, box2.center_y, 1e-6);
   EXPECT_NEAR(0.4, box2.width, 1e-6);
   EXPECT_NEAR(0.5, box2.height, 1e-6);
}

TYPED_TEST(YoloLossLayerTest, TestBoxIOU) {
   typedef typename TypeParam::Dtype Dtype;

   Dtype output_data[5];
   output_data[0] = 0.1;
   output_data[1] = 0.3;
   output_data[2] = 0.3;
   output_data[3] = 0.2;
   output_data[4] = 0.2;
   YoloBox<Dtype> output_box(output_data, false);

   Dtype label_data[5];
   label_data[0] = 3;
   label_data[1] = 0.3;
   label_data[2] = 0.3;
   label_data[3] = 0.2;
   label_data[4] = 0.2;
   YoloBox<Dtype> label_box(label_data, true);
   
   EXPECT_NEAR(1.0, label_box.compute_iou(output_box), 1e-6);
   label_box.center_x = 0.4;
   EXPECT_NEAR(0.50, label_box.compute_iou(output_box), 1e-6);
   label_box.center_x = 0.5;
   EXPECT_NEAR(0.00, label_box.compute_iou(output_box), 1e-6);
}

TYPED_TEST(YoloLossLayerTest, TestBoxFill) {
   typedef typename TypeParam::Dtype Dtype;

   Dtype label_data[5];
   label_data[0] = 10.1;
   label_data[1] = 0.3;
   label_data[2] = 0.3;
   label_data[3] = 0.2;
   label_data[4] = 0.2;
   YoloBox<Dtype> box(label_data, true);
   Dtype expect_label[20];
   box.fill_type_mem(expect_label);
   for (int i = 0; i < 20; i++) {
       if (i != 10) {
           EXPECT_NEAR(0.0, expect_label[i], 1e-6);
       }
   }
   EXPECT_NEAR(1.0, expect_label[10], 1e-6);

   Dtype coord_label[5];
   box.fill_coord_mem(coord_label);
   EXPECT_NEAR(1.0, coord_label[0], 1e-6);
   EXPECT_NEAR(0.3, coord_label[1], 1e-6);
   EXPECT_NEAR(0.3, coord_label[2], 1e-6);
   EXPECT_NEAR(0.2, coord_label[3], 1e-6);
   EXPECT_NEAR(0.2, coord_label[4], 1e-6);
}

}
