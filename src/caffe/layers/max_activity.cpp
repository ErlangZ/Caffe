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

            std::map<int, int> max_values;  //int(max_value * 1e6), max_index

            for (int h = 0; h < height_; h++) {
            for (int w = 0; w < width_; w++) {
                int index = h * width_ + w;
                int key = static_cast<int>(bottom_data[index] * 1e6);
                if (max_values.size() < top_ || max_values.begin()->first < key) {  
                    if (max_values.size() == top_) {
                        max_values.erase(max_values.begin());
                    }
                    max_values.insert(std::make_pair<int, int>(key, index));
                }
            }
            }

            for (int h = 0; h < height_; h++) {
            for (int w = 0; w < width_; w++) {
                int index = h * width_ + w;
                int found = false;
                for (std::map<int, int>::iterator iter = max_values.begin(); 
                        iter != max_values.end(); 
                        iter++) {
                    if (index == iter->second) {
                        found = true;
                    }
                }
                if (found) {
                    top_data[index] = bottom_data[index];
                } else {
                    top_data[index] = 0.0;
                }
            }
            }

            bottom_data += bottom[0]->offset(0, 1);
            top_data += top[0]->offset(0, 1);
        }
    }
}
#ifdef CPU_ONLY
STUB_GPU(MaxActivityLayer);
#endif

INSTANTIATE_CLASS(MaxActivityLayer);
REGISTER_LAYER_CLASS(MaxActivity);


}
