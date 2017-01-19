// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/17 19:15:53
// @file src/caffe/test/test_unpooling_layer.cpp
// @brief 
// 
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UNPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  UNPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()), 
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 2, 2, 4);
    blob_bottom_mask_->Reshape(2, 2, 2, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }
  virtual ~UNPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_mask_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_mask_;
  Blob<Dtype>* const blob_top_;
};

TYPED_TEST_CASE(UNPoolingLayerTest, TestDtypesAndDevices);

// Test for 2x 2 square pooling layer
TYPED_TEST(UNPoolingLayerTest, TestForwardAve) {
    typedef typename TypeParam::Dtype Dtype;
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;

    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
    const int num = 2;
    const int channels = 2;
    this->blob_bottom_->Reshape(num, channels, 2, 4);
    this->blob_bottom_mask_->Reshape(num, channels, 2, 4);
    // Pooling Input: 2x 2 kernel of:                
    //     [1 2 5 2 3] pool [(1+2+9+4)/4, (2+5+4+1)/4, (5+2+1+4)/4, (2+3+4+8)/4] 
    //     [9 4 1 4 8] ->   [(9+4+1+2)/4, (4+1+2+5)/4, (1+4+5+2)/4, (4+8+2+3)/4] 
    //     [1 2 5 2 3]                        
    for (int i = 0; i < 8 * num * channels; i += 8) {
      this->blob_bottom_->mutable_cpu_data()[i +  0] = (1+2+9+4)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  1] = (2+5+4+1)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  2] = (5+2+1+4)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  3] = (2+3+4+8)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  4] = (9+4+1+2)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  5] = (4+1+2+5)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  6] = (1+4+5+2)/4.0;
      this->blob_bottom_->mutable_cpu_data()[i +  7] = (4+8+2+3)/4.0;
    }
    blob_bottom_vec.push_back(this->blob_bottom_);
    blob_top_vec.push_back(this->blob_top_);
 
    UNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    EXPECT_EQ(this->blob_top_->num(), num);
    EXPECT_EQ(this->blob_top_->channels(), channels);
    EXPECT_EQ(this->blob_top_->height(), 3);
    EXPECT_EQ(this->blob_top_->width(), 5);
    
    layer.Forward(blob_bottom_vec, blob_top_vec);
    // [4.0, 3.0, 3.0, 4.25]  -> [4.0, 7.0, 6.0, 7.25, 4.25]
    // [4.0, 3.0, 3.0, 4.25]     [8.0, 14.0, 12.0, 14.5, 8.5]
    //                           [4.0, 7.0, 6.0, 7.25, 4.25]
    //                       mask[1.0, 2.0, 2.0, 2.0, 1.0]
    //                        -> [2.0, 4.0, 4.0, 4.0, 2.0]
    //                           [1.0, 2.0, 2.0, 2.0, 1.0]
    // Expected output: 2x 2 channels of:
    // [4.0, 3.5, 3.0, 3.625, 4.25]
    // [4.0, 3.5, 3.0, 3.625, 4.75]
    // [4.0, 3.5, 3.0, 3.625, 4.25]
    //
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 0], 4.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 1], 3.5, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 2], 3.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 3], 3.625, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 4], 4.25, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 5], 4.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 6], 3.5, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 7], 3.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 8], 3.625, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 9], 4.25, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 10], 4.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 11], 3.5, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 12], 3.0, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 13], 3.625, 1e-6);
      EXPECT_NEAR(this->blob_top_->cpu_data()[i + 14], 4.25, 1e-6);
    }             
}

TYPED_TEST(UNPoolingLayerTest, TestMax) {

    typedef typename TypeParam::Dtype Dtype;
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;

    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    this->blob_bottom_->Reshape(num, channels, 2, 4);
    this->blob_bottom_mask_->Reshape(num, channels, 2, 4);
    // Pooling Input: 2x 2 kernel of:                so Unpooling is
    //     [1 2 5 2 3] pool [9, 5, 5, 8]             [0, 0, 5, 0, 0]
    //     [9 4 1 4 8] ->   [9, 5, 5, 8]   unpooling [9, 0, 0, 0, 8]
    //     [1 2 5 2 3]                         ->    [0, 0, 5, 0, 0]
    //                 mask [5, 2, 2, 9]
    //                 ->   [5, 12, 12, 9] 
    //
    for (int i = 0; i < 8 * num * channels; i += 8) {
      this->blob_bottom_->mutable_cpu_data()[i +  0] = 9;
      this->blob_bottom_->mutable_cpu_data()[i +  1] = 5;
      this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      this->blob_bottom_->mutable_cpu_data()[i +  3] = 8;
      this->blob_bottom_->mutable_cpu_data()[i +  4] = 9;
      this->blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      this->blob_bottom_->mutable_cpu_data()[i +  6] = 5;
      this->blob_bottom_->mutable_cpu_data()[i +  7] = 8;
    
      this->blob_bottom_mask_->mutable_cpu_data()[i +  0] = 5;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  1] = 2;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  2] = 2;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  3] = 9;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  4] = 5;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  5] = 12;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  6] = 12;
      this->blob_bottom_mask_->mutable_cpu_data()[i +  7] = 9;
    }
    blob_bottom_vec.push_back(this->blob_bottom_);
    blob_bottom_vec.push_back(this->blob_bottom_mask_);
    blob_top_vec.push_back(this->blob_top_);
    
    UNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    EXPECT_EQ(this->blob_top_->num(), num);
    EXPECT_EQ(this->blob_top_->channels(), channels);
    EXPECT_EQ(this->blob_top_->height(), 3);
    EXPECT_EQ(this->blob_top_->width(), 5);
    
    layer.Forward(blob_bottom_vec, blob_top_vec);
    // Expected output: 2x 2 channels of:
    // [0, 0, 5, 0, 0]
    // [9, 0, 0, 0, 8]
    // [0, 0, 5, 0, 0]
    //
    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 9);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 8);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 5);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 0);
    }           

}

}
