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
    double* cv_data = cv_data_.mutable_cpu_data();
    for (int num = 0; num < bottom[0]->num(); num++) {
        cv::Mat image;
        if (bottom[0]->channels() == 1) {
            image = cv::Mat(bottom[0]->height(), bottom[0]->width(), CV_64FC1, image_data); 
        } else if (bottom[0]->channels() == 3) {
            for (int h = 0; h < bottom[0]->height(); h++) {
                for (int w = 0; w < bottom[0]->width(); w++) {
                    for (int c = 0; c < bottom[0]->channels(); c++) {
                        cv_data[h * (channels_ * width_) + w * channels_ + c] =
                            image_data[c * (height_ * width_) + h * width_ + w];
                    }
                }
            }
            image = cv::Mat(height_, width_, CV_64FC3, cv_data); 
        } else {
            LOG(FATAL) << "Get Invalid Image Channels:" << bottom[0]->channels();
        }
        cv::imshow(picture_name.c_str(), image);
        LOG(INFO) << "Show:" << picture_name << " num:" << num;
        cv::waitKey(0);
        image_data += bottom[0]->offset(0);
    }
}

template<>
void ShowImageLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom, 
                                    const vector<Blob<float>*>& top) {
    float* image_data = bottom[0]->mutable_cpu_data();
    float* cv_data = cv_data_.mutable_cpu_data();
    for (int num = 0; num < bottom[0]->num(); num++) {
        cv::Mat image;
        if (bottom[0]->channels() == 1) {
            image = cv::Mat(bottom[0]->height(), bottom[0]->width(), CV_32FC1, image_data); 
        } else if (bottom[0]->channels() == 3) {
            for (int h = 0; h < bottom[0]->height(); h++) {
                for (int w = 0; w < bottom[0]->width(); w++) {
                    for (int c = 0; c < bottom[0]->channels(); c++) {
                        cv_data[h * (channels_ * width_) + w * channels_ + c] =
                            image_data[c * (height_ * width_) + h * width_ + w];
                    }
                }
            }
            image = cv::Mat(height_, width_, CV_32FC3, cv_data); 
        } else {
            LOG(FATAL) << "Get Invalid Image Channels:" << bottom[0]->channels();
        }
        cv::imshow(picture_name.c_str(), image);
        LOG(INFO) << "Show:" << picture_name << " num:" << num;
        cv::waitKey(0);
        image_data += bottom[0]->offset(0);
    }
}

#ifdef CPU_ONLY                                                                                     
STUB_GPU(ShowImageLayer);                                                                           
#endif                                                                                              
INSTANTIATE_CLASS(ShowImageLayer);                                                                  
REGISTER_LAYER_CLASS(ShowImage);   
}
