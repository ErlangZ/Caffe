// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/19 19:24:02
// @file src/caffe/layers/show_image_layer.cpp
// @brief 
// 

#include "caffe/layers/show_image_layer.hpp"

namespace caffe {
template<>
void ShowImageLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom, 
                                    const vector<Blob<double>*>& top) {
    double* image_data = bottom[0]->mutable_cpu_data();
    for (int num = 0; num < bottom[0]->num(); num++) {
        for (int channel = 0; channel < bottom[0]->channel(); channel++) {
            cv::Mat image;
            image = cv::Mat(bottom[0]->height(), bottom[0]->width(), CV_64FC1,
                                    image_data); 
            cv::imshow(picture_name, image);
            cv::waitKey(0);
        } 
    }
}

template<>
void ShowImageLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom, 
                                    const vector<Blob<float>*>& top) {
    float* image_data = bottom[0]->mutable_cpu_data();
    for (int num = 0; num < bottom[0]->num(); num++) {
        cv::Mat image;
        if (bottom[0]->channels() == 1) {
            image = cv::Mat(bottom[0]->height(), bottom[0]->width(), CV_32FC1,
                                    image_data); 
        } else if (bottom[0]->channels() == 3) {
            image = cv::Mat(bottom[0]->height(), bottom[0]->width(), CV_32FC3,
                                    image_data); 
        } else {
            LOG(FATAL) << "Get Invalid Image Channels:" << bottom[0]->channels();
        }
        cv::imshow(picture_name, image);
        cv::waitKey(0);
    }
}

#ifdef CPU_ONLY                                                                                     
STUB_GPU(ShowImageLayer);                                                                           
#endif                                                                                              
INSTANTIATE_CLASS(ShowImageLayer);                                                                  
REGISTER_LAYER_CLASS(ShowImage);   
}
