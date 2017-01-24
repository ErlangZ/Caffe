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
    
    int num = std::min(MAX_NUM, static_cast<int>(std::floor(std::sqrt(num_))));

    float* image_data = bottom[0]->mutable_cpu_data();

    int offset = 0;
    if (false) {
        caffe_sub(mean_data_.count(), image_data + offset, mean_data_.cpu_data(), image_data + offset);     
        offset += height_ * width_;                                  
        caffe_sub(mean_data_.count(), image_data + offset, mean_data_.cpu_data(), image_data + offset);     
        offset += height_ * width_;                                  
        caffe_sub(mean_data_.count(), image_data + offset, mean_data_.cpu_data(), image_data + offset);     
    }
    cv::Mat image1 = cv::Mat(height_, width_, CV_32FC1, image_data);
//    normalize(image1,image1,255.0,0.0,cv::NORM_MINMAX);
    cv::Mat image2 = cv::Mat(height_, width_, CV_32FC1, image_data + height_ * width_);
//    normalize(image2,image2,255.0,0.0,cv::NORM_MINMAX);
    cv::Mat image3 = cv::Mat(height_, width_, CV_32FC1, image_data + 2 * height_ * width_);
//    normalize(image3,image3,255.0,0.0,cv::NORM_MINMAX);
    std::string name1("image1");
    std::string name2("image2");
    std::string name3("image3");
    cv::imshow((picture_name+name1).c_str(), image1);
    cv::imshow((picture_name+name2).c_str(), image2);
    cv::imshow((picture_name+name3).c_str(), image3);
    if (wait_) cv::waitKey(0);
}

#ifdef CPU_ONLY                                                                                     
STUB_GPU(ShowImageLayer);                                                                           
#endif                                                                                              
INSTANTIATE_CLASS(ShowImageLayer);                                                                  
REGISTER_LAYER_CLASS(ShowImage);   
}
