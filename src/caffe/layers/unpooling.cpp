// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/16 13:57:04
// @file src/caffe/layers/unpooling.cpp
// @brief 
// 

#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/util/device_alternate.hpp"
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;
 
template <typename Dtype>
void UNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  CHECK(!pool_param.has_global_pooling()) << "UNPoolingLayer should not be global_pooling.";
  CHECK(!pool_param.has_kernel_size() != !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(pool_param.has_kernel_size() || (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
        << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK(!(pool_param.has_pad() && pool_param.has_pad_h() && pool_param.has_pad_w()))
        << "pad is pad OR pad_h and pad_w should not be presented.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";

  if (pool_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = pool_param.kernel_size();
  } else {
    kernel_h_ = pool_param.kernel_h();
    kernel_w_ = pool_param.kernel_w();
  }
  
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
}

template <typename Dtype>
void UNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  pooled_height_ = (height_ - 1) * stride_h_ + kernel_h_;
  pooled_width_ = (width_ - 1) * stride_w_ + kernel_w_;
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_AVE && top.size() == 1) {
      ave_count_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void UNPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mask_data = NULL;
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  Dtype* count_data = NULL;

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    mask_data = bottom[1]->cpu_data();
    caffe_set(top_count, static_cast<Dtype>(0.0), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          for (int pw = 0; pw < width_; ++pw) {
            int index = ph * height_ + pw;
            int max_pool_index = static_cast<int>(mask_data[index]);
            top_data[max_pool_index] = bottom_data[index]; 
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        mask_data += bottom[1]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
    case PoolingParameter_PoolMethod_AVE:
    count_data = ave_count_.mutable_cpu_data();
    // The main loop
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
      count_data[i] = 0;
    }
    
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < height_; ++ph) {
          for (int pw = 0; pw < width_; ++pw) {
            int index = ph * width_ + pw;
            int hstart = ph * stride_h_;
            int wstart = pw * stride_w_;
            int hend = min(hstart + kernel_h_, pooled_height_);
            int wend = min(wstart + kernel_w_, pooled_width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[h * pooled_width_ + w] += bottom_data[index]; 
                count_data[h * pooled_width_ + w] += 1;
              }
            }
          }
        }
        //get mean.
        for (int i = 0; i < pooled_height_ * pooled_width_; i++) {
            if (count_data[i] > 0) {
                top_data[i] /= count_data[i];
            }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        count_data += ave_count_.offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
};

#ifdef CPU_ONLY
STUB_GPU(UNPoolingLayer);
#endif

INSTANTIATE_CLASS(UNPoolingLayer);


}

