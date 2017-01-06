// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/01/05 17:29:46
// @file src/caffe/layers/pcd_data_layer.cpp
// @brief 

#include "caffe/layers/pcd_data_layer.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
PCDDataLayer<Dtype>::~PCDDataLayer<Dtype>() {
    this->StopInternalThread();
}


template <typename Dtype>
void PCDDataLayer<Dtype>::DataLayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                                         const std::vector<Blob<Dtype>*>& top) {

    CHECK(_label_reader.init(this->layer_param_.pcd_data_param().root_folder()));
}

// This function is called on prefetch thread
template <typename Dtype>
void PCDDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

}

INSTANTIATE_CLASS(PCDDataLayer);
REGISTER_LAYER_CLASS(PCDData);

}
