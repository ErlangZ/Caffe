// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/02/27 10:52:57
// @file src/caffe/layers/yolo_pre_train_loss.cpp
// @brief 
// 
#include "caffe/layers/yolo_pre_train_loss.hpp"
namespace caffe {

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {

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
            loss -= target_data[c] * log(output_data[0]) + 
                    (1 - target_data[c]) * log(1 - output_data[0]);
        }
        output_data += bottom[i]->count(1);
    }
    loss /= target_layer->num();
}

template<typename Dtype>
void YoloPretrainedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<bool>& propagate_down,
                                                  const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* target_layer = bottom[0];
    const Dtype* target_data = target_layer->cpu_data();
    for (int i = 1; i < bottom.size(); i++)  {
        const Dtype* output_data = bottom[i]->cpu_data();
        for (int c = 0; c < target_layer->channels(); c++) {
            Dtype diff = (1 - target_data[c]) / (1 - output_data[0]) - target_data[c] / output_data[0];
            bottom[i]->mutable_cpu_diff()[c] = diff;
        }
        output_data += bottom[i]->count(1);
    }
}
#ifdef CPU_ONLY                                                                                     
STUB_GPU(YoloPretrainedLossLayer);                                                                       
#endif                                                                                              
                                                                                                      
INSTANTIATE_CLASS(YoloPretrainedLossLayer);                                                             
REGISTER_LAYER_CLASS(YoloPretrainedLoss);     

} // namespace 
