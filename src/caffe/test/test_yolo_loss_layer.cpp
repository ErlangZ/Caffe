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
#include "caffe/layers/yolo_data_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class YoloLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  YoloLossLayerTest<TypeParam>() :
      //Data:
      //data N = 2, S_ = 7; B_ = 2; class_number = 4
      //output N * S * S * (2 * 5 + 4)
      //Label:
      blob_bottom_data_(new Blob<Dtype>(2, 686, 1, 1)),
      blob_bottom_label_(new Blob<Dtype>(2, 51, 1, 1)), 
      blob_top_loss_(new Blob<Dtype>()) {

      fill_label();

      // fill the values                                                                              
      blob_bottom_vec_.push_back(blob_bottom_data_);                                                  
      blob_bottom_vec_.push_back(blob_bottom_label_);                                                 
      blob_top_vec_.push_back(blob_top_loss_);     
  }
  virtual ~YoloLossLayerTest<TypeParam>() {
      delete blob_bottom_data_;                                                                       
      delete blob_bottom_label_;                                                                      
      delete blob_top_loss_;    
  }
 protected:
  void fill_label() {
      Dtype* label = blob_bottom_label_->mutable_cpu_data();
      //batch = 1
      label[0] = 1.1; //there is only one object.
      label[1] = 2; // type = 2
      label[2] = 0.2; //center_x = 0.2
      label[3] = 0.2; //center_y = 0.2
      label[4] = 0.1; //width = 0.1
      label[5] = 0.1; //height = 0.1
      //batch = 2
      label += blob_bottom_label_->count(1);
      label[0] = 2.1; //there is 2 objects.
      label[1] = 1; // type = 1
      label[2] = 0.3; //center_x = 0.3
      label[3] = 0.3; //center_y = 0.3
      label[4] = 0.1; //width = 0.1
      label[5] = 0.1; //height = 0.1
      label[6] = 3; // type = 3
      label[7] = 0.7; //center_x = 0.7
      label[8] = 0.7; //center_y = 0.7
      label[9] = 0.1; //width = 0.1
      label[10] = 0.1; //height = 0.1
  }
 protected:
  Blob<Dtype>* const blob_bottom_data_;                                                             
  Blob<Dtype>* const blob_bottom_label_;                                                            
  Blob<Dtype>* const blob_top_loss_;                                                                
  vector<Blob<Dtype>*> blob_bottom_vec_;                                                            
  vector<Blob<Dtype>*> blob_top_vec_;    
};

TYPED_TEST_CASE(YoloLossLayerTest, TestDtypesAndDevices);

/*
TYPED_TEST(YoloLossLayerTest, ReadData) {
    typedef typename TypeParam::Dtype Dtype;

    vector<Blob<Dtype>*> blob_bottom_vec;                                                            
    vector<Blob<Dtype>*> blob_top_vec(2, NULL);    
    blob_top_vec[0] = new Blob<Dtype>();
    blob_top_vec[1] = new Blob<Dtype>();

    LayerParameter layer_param;    
    //std::string db_file(CMAKE_SOURCE_DIR "/caffe/test/test_data/yolo_train_lmdb");
    std::string db_file("/home/erlangz/Caffe/models/Pascal/pascal_train_lmdb");
    layer_param.mutable_data_param()->set_source(db_file);
    layer_param.mutable_data_param()->set_batch_size(1);
    layer_param.mutable_yolo_data_param()->set_max_labels_number(10); //10 * 5 + 1
    layer_param.mutable_data_param()->set_backend(DataParameter::LMDB);

    YoloDataLayer<Dtype> data_layer(layer_param);
    data_layer.LayerSetUp(blob_bottom_vec, blob_top_vec);
    while(true) {
       data_layer.Forward(blob_bottom_vec, blob_top_vec);
       std::vector<YoloBox<Dtype> > labels_boxes;
       build_boxes_from_labels(blob_top_vec[1]->cpu_data(), &labels_boxes);
       for (int i = 0; i < labels_boxes.size(); i++) {
       //if (labels_boxes[0].width < 0.0 || labels_boxes[0].height < 0.0) {
         std::cout << labels_boxes[i].center_x << " " << labels_boxes[i].center_x<< " "
                   << labels_boxes[i].width << " " << labels_boxes[i].height << " " 
                  << labels_boxes[i].type << std::endl;
       // }
       }
    }
}
*/

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

TYPED_TEST(YoloLossLayerTest, TestReshape) {
   typedef typename TypeParam::Dtype Dtype;
   LayerParameter layer_param;    
   layer_param.mutable_yolo_loss_param()->set_class_number(4);
   YoloLossLayer<Dtype> loss_layer(layer_param);
   loss_layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
   EXPECT_EQ(1, this->blob_top_vec_[0]->count());
}

TYPED_TEST(YoloLossLayerTest, TestForwardBackWard) {
   typedef typename TypeParam::Dtype Dtype;
   LayerParameter layer_param;    
   layer_param.mutable_yolo_loss_param()->set_class_number(4);
   layer_param.mutable_yolo_loss_param()->set_iou_threshold(0.01);
   YoloLossLayer<Dtype> loss_layer(layer_param);
   loss_layer.lambda_coord_ = 5.0;
   loss_layer.lambda_noobj_ = 0.5;
   loss_layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
   ASSERT_EQ(1, this->blob_top_vec_[0]->count());

   //Set Data.
   Dtype* data = this->blob_bottom_data_->mutable_cpu_data();
   caffe_memset(sizeof(Dtype) * (this->blob_bottom_data_->count()), 0, data);
   //Batch 0 
   Dtype* now_data = data + loss_layer.get_offset(1, 1); 
   now_data[0] = 0.4;  //confidence = 0.4
   now_data[1] = 0.2;  //center_x = 0.2
   now_data[2] = 0.2;  //center_y = 0.2
   now_data[3] = 0.14;  //width = 0.1
   now_data[4] = 0.08;  //height = 0.1
   now_data[5] = 0.3;  //confidence = 0.3
   now_data = data + loss_layer.get_offset(1, 1); //(0.2, 0.2) in (1, 1) 
   now_data[11] = 0.4;   //type 1 poss
   now_data[12] = 1.4;   //type 2 poss(right)
   //Batch 1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(2, 3);
   now_data[0] = 0.4;  //confidence = 0.4
   now_data[1] = 0.3;  //center_x = 0.3
   now_data[2] = 0.3;  //center_y = 0.3
   now_data[3] = 0.14;  //width = 0.1
   now_data[4] = 0.08;  //height = 0.1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(2, 2);
   now_data[11] = 0.4;   //type 1 poss(right)
   now_data[12] = 1.6;   //type 2 poss

   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(5, 6);
   now_data[0] = 0.4;  //confidence = 0.4
   now_data[1] = 0.7;  //center_x = 0.7
   now_data[2] = 0.7;  //center_y = 0.7
   now_data[3] = 0.14;  //width = 0.1
   now_data[4] = 0.08;  //height = 0.1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(5, 5);
   now_data[11] = 0.4;   //type 1 poss
   now_data[13] = 1.7;   //type 3 poss(right)

   //Foward
   loss_layer.Forward_cpu(this->blob_bottom_vec_, this->blob_top_vec_);
   loss_layer.Backward_cpu(this->blob_top_vec_, vector<bool>(), this->blob_bottom_vec_);

   //EXPECT
   //batch-0 label-0
   EXPECT_NEAR((0.4 - 1.0) * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[0], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_* 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[1], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_* 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[2], 1e-4);
   EXPECT_NEAR(0.0774228 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[3], 1e-4);
   EXPECT_NEAR(-0.059016 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[4], 1e-4);
   EXPECT_NEAR((0.3 - 0.0) * loss_layer.lambda_noobj_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[5], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[6], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[7], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[8], 1e-4);
   EXPECT_NEAR(0.0 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[9], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[10], 1e-4);
   EXPECT_NEAR(0.4 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[11], 1e-4);
   EXPECT_NEAR(0.4 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[12], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + loss_layer.get_offset(1, 1))[13], 1e-4);

   //batch-1 label-0
   EXPECT_NEAR((0.4 - 1.0) * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[0], 1e-4);
   EXPECT_NEAR(0.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1)  + loss_layer.get_offset(2, 3))[1], 1e-4);
   EXPECT_NEAR(0.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[2], 1e-4);
   EXPECT_NEAR(0.0774228 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[3], 1e-4);
   EXPECT_NEAR(-0.059016 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[4], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[10], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[11], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[12], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 3))[13], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 2))[10], 1e-4);
   EXPECT_NEAR((0.4 - 1.0) * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 2))[11], 1e-4);
   EXPECT_NEAR((1.6 - 0.0) * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 2))[12], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(2, 2))[13], 1e-4);

   //batch-1 label-1
   EXPECT_NEAR((0.4 - 1.0) * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 6))[0], 1e-4);
   EXPECT_NEAR(0.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1)  + loss_layer.get_offset(5, 6))[1], 1e-4);
   EXPECT_NEAR(0.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 6))[2], 1e-4);
   EXPECT_NEAR(0.0774228 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 6))[3], 1e-4);
   EXPECT_NEAR(-0.059016 * loss_layer.lambda_coord_ * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 6))[4], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 5))[10], 1e-4);
   EXPECT_NEAR((0.4 - 0.0) * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 5))[11], 1e-4);
   EXPECT_NEAR(0.0 * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 5))[12], 1e-4);
   EXPECT_NEAR((1.7 - 1.0) * 1.0 * 2.0, (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1) + loss_layer.get_offset(5, 5))[13], 1e-4);
/*
   for (int i = 0; i < this->blob_bottom_data_->count(); i++) {
       std::cout << this->blob_bottom_data_->cpu_data()[i] << " ";
   }
   std::cout << endl;
   for (int i = 0; i < this->blob_bottom_data_->count(); i++) {
       std::cout << this->blob_bottom_data_->cpu_diff()[i] << " ";
   }
   std::cout << endl;
*/
}

TYPED_TEST(YoloLossLayerTest, TestForwardBackWard2) {
   typedef typename TypeParam::Dtype Dtype;
   LayerParameter layer_param;    
   layer_param.mutable_yolo_loss_param()->set_class_number(4);
   layer_param.mutable_yolo_loss_param()->set_iou_threshold(0.40);
   YoloLossLayer<Dtype> loss_layer(layer_param);
   loss_layer.lambda_coord_ = 5.0;
   loss_layer.lambda_noobj_ = 0.5;
   loss_layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
   ASSERT_EQ(1, this->blob_top_vec_[0]->count());

   //Set Data.
   Dtype* data = this->blob_bottom_data_->mutable_cpu_data();
   caffe_memset(sizeof(Dtype) * (this->blob_bottom_data_->count()), 0, data);
   //Batch 0 
   Dtype* now_data = data + loss_layer.get_offset(1, 1); 
   now_data[0] = 1.0;  //confidence = 0.4
   now_data[1] = 0.2;  //center_x = 0.2
   now_data[2] = 0.2;  //center_y = 0.2
   now_data[3] = 0.1;  //width = 0.1
   now_data[4] = 0.1;  //height = 0.1
   now_data[5] = 0.0;  //confidence = 0.0
   now_data = data + loss_layer.get_offset(1, 1); //(0.2, 0.2) in (1, 1) 
   now_data[12] = 1.0;   //type 2 poss(right)
   //Batch 1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(2, 3);
   now_data[0] = 1.0;  //confidence = 0.4
   now_data[1] = 0.3;  //center_x = 0.3
   now_data[2] = 0.3;  //center_y = 0.3
   now_data[3] = 0.1;  //width = 0.1
   now_data[4] = 0.1;  //height = 0.1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(2, 2);
   now_data[11] = 1.0;   //type 1 poss(right)

   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(5, 6);
   now_data[0] = 1.0;  //confidence = 0.4
   now_data[1] = 0.7;  //center_x = 0.7
   now_data[2] = 0.7;  //center_y = 0.7
   now_data[3] = 0.1;  //width = 0.1
   now_data[4] = 0.1;  //height = 0.1
   now_data = data + this->blob_bottom_data_->count(1) + loss_layer.get_offset(5, 4);
   now_data[13] = 1.0; //type 3 poss(right)

   //Foward
   loss_layer.Forward_cpu(this->blob_bottom_vec_, this->blob_top_vec_);
   loss_layer.Backward_cpu(this->blob_top_vec_, vector<bool>(), this->blob_bottom_vec_);

   //EXPECT
   EXPECT_NEAR(0.0, this->blob_top_vec_[0]->cpu_data()[0], 1e-6);
   /*
   std::cout << "XXXXXXXXXXXXXXXXXXXX" << endl;
   for (int i = 0; i < this->blob_bottom_data_->count(); i++) {
       std::cout << this->blob_bottom_data_->cpu_data()[i] << " ";
   }
   std::cout << endl;
   std::cout << "XXXXXXXXXXXXXXXXXXXX" << endl;
   for (int i = 0; i < loss_layer.S_; i++) {
   for (int j = 0; j < loss_layer.S_; j++) {
       std::cout << "i:" << i << " j:" << j << std::endl;
       for (int k = 0; k < loss_layer.count_; k++) {
          std::cout << (this->blob_bottom_data_->cpu_diff() + this->blob_bottom_data_->offset(1))[loss_layer.get_offset(i, j) + k] << " ";
       }
       std::cout << std::endl;
   }
   }
   std::cout << endl;
   */
}
}
