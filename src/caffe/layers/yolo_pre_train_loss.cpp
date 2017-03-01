// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/02/27 10:52:57
// @file src/caffe/layers/yolo_pre_train_loss.cpp
// @brief 
// 
#include "caffe/layers/yolo_pre_train_loss.hpp"
#include <float.h>
namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* target_layer = bottom[0];
    CHECK_EQ(bottom.size(), target_layer->channels() + 1) 
        << "YoloPretrainedLossLayer's Input Label's Channel must be equal to Output Channels";
    for (int i = 0; i < bottom.size(); i++) {
        CHECK_EQ(target_layer->num(), bottom[i]->num()) 
            << "YoloPretrainedLossLayer's num must be equal";
        CHECK_EQ(1, bottom[i]->height()) << "YoloPretrainedLossLayer's bottom must be [1X1]";
        CHECK_EQ(1, bottom[i]->width()) << "YoloPretrainedLossLayer's bottom must be [1X1]";
    }
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    Dtype loss = Dtype(0.0);
    Blob<Dtype>* target_layer = bottom[0];
    const Dtype* target_data = target_layer->cpu_data();
    for (int i = 1; i < bottom.size(); i++)  {
        const Dtype* input_data = bottom[i]->cpu_data();
        const int c = i - 1;
        for (int n = 0; n < target_layer->num(); n++) {
            const Dtype& y = target_data[n * target_layer->num() + c];
            const Dtype& x = input_data[n];
            loss -= x * (y - (x>=0)) - log(1 + exp(x - 2 * x * (x>=0)));
            //std::cout << "Channel:" << c << " loss:" << loss << " target_data:" << y << " input_data: "<< x << std::endl;
        }
    }
    loss /= target_layer->count();
    top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {
    // Top & Bottom is inversed with Forward_cpu
    // bottom [0] is label, 
    // bottom [1 ... c+1) are input
    Blob<Dtype>* target_layer = bottom[0];
    const Dtype* target_data = target_layer->cpu_data();
    for (int i = 0; i < bottom.size(); i++)  {
        if (!propagate_down[i]) { 
            //LOG(INFO) << "Bottom layer Backward:" << i << " skiped.";
            continue; 
        }
        const int c = i - 1;
        const Dtype* input_data = bottom[i]->cpu_data(); // the (i-1)th channel with num values. 
        for (int n = 0; n < target_layer->num(); n++) {
            const Dtype& y = target_data[n * target_layer->num() + c];
            const Dtype& h = sigmoid(input_data[n]);
            bottom[i]->mutable_cpu_diff()[n] = h - y;
            //std::cout << "Channel:" << c << " diff:" << h - y << " target_data:" << y << " input_data: "<< h << std::endl;
        }
    }
}

#ifdef CPU_ONLY                                                                                     
STUB_GPU(YoloPretrainedLossLayer);                                                                       
#endif                                                                                              
                                                                                                      
INSTANTIATE_CLASS(YoloPretrainedLossLayer);                                                             
REGISTER_LAYER_CLASS(YoloPretrainedLoss);     

} // namespace 
