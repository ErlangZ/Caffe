// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/19 19:24:02
// @file src/caffe/layers/show_image_layer.cpp
// @brief 
// 

#include "caffe/layers/show_image_layer.hpp"
const int MAX_NUM = 10;
namespace caffe {
template<>
void ShowImageLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom, 
                                    const vector<Blob<double>*>& top) {
    NOT_IMPLEMENTED;
    /*
    int num = std::min(MAX_NUM, num_);
    double* image_data = bottom[0]->mutable_cpu_data();
    double* cv_data = cv_data_;
    for (int n = 0; n < num; n++) {
        for (int h = 0; h < bottom[0]->height(); h++) {
            for (int w = 0; w < bottom[0]->width(); w++) {
                for (int c = 0; c < bottom[0]->channels(); c++) {
                    cv_data[h * (channels_ * width_ * num) + w * channels_ + c] = 
                            image_data[c * (height_ * width_) + h * width_ + w];
                }
            }
        }
        image_data += bottom[0]->offset(1);
        cv_data += bottom[0]->offset(1) * num;
    }
    cv::Mat image;
    if (bottom[0]->channels() == 1) {
        image = cv::Mat(height_ * num, width_ * num, CV_64FC1, cv_data_); 
    } else if (bottom[0]->channels() == 3) {
        image = cv::Mat(height_ * num, width_ * num, CV_64FC3, cv_data_); 
    } else {
        LOG(FATAL) << "Get Invalid Image Channels:" << bottom[0]->channels();
    }
    cv::imshow(picture_name.c_str(), image);
    if (wait_) cv::waitKey(0);
    */
}

template<>
void ShowImageLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom, 
                                        const vector<Blob<float>*>& top) {
    
    int num = std::min(MAX_NUM, num_);
    float* image_data = bottom[0]->mutable_cpu_data();
    float* cv_data = cv_data_;
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < bottom[0]->channels(); c++) {
            float min_value = FLT_MAX;
            float max_value = -FLT_MAX;
            if (normalize_) {
                for (int h = 0; h < bottom[0]->height(); h++) {
                    for (int w = 0; w < bottom[0]->width(); w++) {
                        float v = image_data[c * (height_ * width_) + h * width_ + w];
                        if (v < min_value) {
                            min_value = v;
                        }
                        if (v > max_value) {
                            max_value = v;
                        }
                    }
                }
            }
            for (int h = 0; h < bottom[0]->height(); h++) {
                for (int w = 0; w < bottom[0]->width(); w++) {
                    if (!normalize_) {
                        cv_data[h * (channels_ * width_ * num) + w * channels_ + c] = 
                                scale_ * image_data[c * (height_ * width_) + h * width_ + w];
                    } else {
                        cv_data[h * (channels_ * width_ * num) + w * channels_ + c] = 
                                scale_ * (image_data[c * (height_ * width_) + h * width_ + w] - min_value)/(max_value - min_value) * 255.0;

                    }
                }
            }
        }         
        image_data += bottom[0]->offset(1);
        cv_data += bottom[0]->offset(1) * num;
    }
    cv::Mat image;
    if (bottom[0]->channels() == 1) {
        image = cv::Mat(height_ * num, width_ * num, CV_32FC1, cv_data_); 
    } else if (bottom[0]->channels() == 3) {
        image = cv::Mat(height_ * num, width_ * num, CV_32FC3, cv_data_); 
    } else {
        LOG(FATAL) << "Get Invalid Image Channels:" << bottom[0]->channels();
    }
    cv::imshow(picture_name.c_str(), image);
    if (wait_) cv::waitKey(0);
}

#ifdef CPU_ONLY                                                                                     
STUB_GPU(ShowImageLayer);                                                                           
#endif                                                                                              
INSTANTIATE_CLASS(ShowImageLayer);                                                                  
REGISTER_LAYER_CLASS(ShowImage);   
}
