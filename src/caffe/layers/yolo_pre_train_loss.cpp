// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/02/27 10:52:57
// @file src/caffe/layers/yolo_pre_train_loss.cpp
// @brief 
// 
#include "caffe/layers/yolo_pre_train_loss.hpp"
#include <float.h>
namespace caffe {

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
        const Dtype* output_data = bottom[i]->cpu_data();
        for (int c = 0; c < target_layer->channels(); c++) {
            loss -= target_data[c] * log(0.0001 + output_data[c]) + (1 - target_data[c]) * log(1.0001 - output_data[c]);
            //std::cout << "Channel:" << c << " loss:" << loss << " target_data:" << target_data[c] << " output_data: "<< output_data[c] << std::endl;
        }
        target_data += target_layer->count(1);
    }
    loss /= target_layer->num();
    top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& bottom) {
//    std::cout << "top size:" << top.size() << " shape:" << top[0]->shape_string() << std::endl;
//    std::cout << "bottom size:" << bottom.size() << " shape:" << bottom[0]->shape_string() << std::endl;
    // Top & Bottom is inversed with Forward_cpu
    Blob<Dtype>* target_layer = bottom[0];
    const Dtype* target_data = target_layer->cpu_data();
    for (int i = 0; i < bottom.size(); i++)  {
        if (!propagate_down[i]) continue; 
        const Dtype* output_data = bottom[i]->cpu_data();
/*
        std::cout << "OutputData:" << "num:" << i << std::endl;
        for (int j = 0; j < 20; j++) {
            std::cout << output_data[j] << " ";
        }
        std::cout << std::endl; 
*/
        for (int c = 0; c < target_layer->channels(); c++) {
            Dtype diff = 0.0;
            /*if (fabs(output_data[c]) < 1e-4) {
                diff = -FLT_MAX; 
            } else if(fabs(1 - output_data[c]) < 1e-4) {
                diff = FLT_MAX; 
            } else {
                diff = (1 - target_data[c]) / (1 - output_data[c]) - (target_data[c] / (output_data[c]));
            }
            */
            diff = (1 - target_data[c]) / (1.0001 - output_data[c]) - (target_data[c] / (0.0001 + output_data[c]));
            //std::cout << "Channel:" << c << " Diff:" << diff << std::endl;
            bottom[c]->mutable_cpu_diff()[i] = diff;
        }
        target_data += target_layer->count(1);
    }
}
#ifdef CPU_ONLY                                                                                     
STUB_GPU(YoloPretrainedLossLayer);                                                                       
#endif                                                                                              
                                                                                                      
INSTANTIATE_CLASS(YoloPretrainedLossLayer);                                                             
REGISTER_LAYER_CLASS(YoloPretrainedLoss);     

} // namespace 
