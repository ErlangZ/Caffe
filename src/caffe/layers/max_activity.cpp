// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/22 17:29:56
// @file src/caffe/layers/max_activity.cpp
// @brief 
// 
#include "caffe/layers/max_activity.hpp"
#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/util/device_alternate.hpp"
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype>
void MaxActivityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                          const vector<Blob<Dtype>*>& top) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    
    for (int n = 0; n < num_; n++) {
        for (int c = 0; c < channels_; c++) {
            Dtype max_value = -FLT_MAX; 
            int max_h = -1;
            int max_w = -1;
            for (int h = 0; h < height_; h++) {
            for (int w = 0; w < width_; w++) {
                int index = h * width_ + w;
                if (bottom_data[index] > max_value) {
                    max_value = bottom_data[index];
                    max_h = h;
                    max_w = w;
                }
            }
            }

            for (int h = 0; h < height_; h++) {
            for (int w = 0; w < width_; w++) {
                int index = h * width_ + w;
                if (max_h == h && max_w == w) {
                    top_data[index] = bottom_data[index];
                } else {
                    top_data[index] = 0.0;
                }
            }
            }
        }
    }
}
#ifdef CPU_ONLY
STUB_GPU(MaxActivityLayer);
#endif

INSTANTIATE_CLASS(MaxActivityLayer);
REGISTER_LAYER_CLASS(MaxActivity);


}
