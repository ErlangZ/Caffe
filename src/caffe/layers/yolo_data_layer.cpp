// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/03/09 15:43:37
// @file src/caffe/layers/yolo_data_layer.cpp
// @brief 
// 
#include <vector>

#include "caffe/layers/yolo_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
YoloDataLayer<Dtype>::YoloDataLayer(const LayerParameter& param) : 
    BasePrefetchingDataLayer<Dtype>(param), 
    reader_(param) {
}
template <typename Dtype>
YoloDataLayer<Dtype>::~YoloDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void YoloDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                          const vector<Blob<Dtype>*>& top) {

  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

  // Each Image has at most MAX_LABEL labels
  // label's first element is the number of labels.
  // Each label has 5 column class/x_center/y_center/width/height
  const int max_labels_number = this->layer_param_.yolo_data_param().max_labels_number();
  if (this->output_labels_) {
    vector<int> label_shape(4, 1);
    label_shape[0] = batch_size;
    label_shape[1] = 1 + max_labels_number * 5;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
         this->prefetch_[i].label_.Reshape(label_shape);
    }
    top[1]->Reshape(label_shape);
  }
}


// This function is called on prefetch thread
template<typename Dtype>
void YoloDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int data_offset = batch->data_.offset(item_id);
    int label_offset = batch->label_.offset(item_id); 
    this->transformed_data_.set_cpu_data(top_data + data_offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
        init_yolo_label(top_label + label_offset, datum); 
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

template<typename Dtype>
void YoloDataLayer<Dtype>::init_yolo_label(Dtype* label, const Datum& datum) {
    label[0] = datum.yolo_labels_size();
    for (int i = 0; i < datum.yolo_labels_size(); i++) {
        label[i * 5 + 1] = datum.yolo_labels(i).label(); 
        const Dtype x_min = datum.yolo_labels(i).x_min();
        const Dtype y_min = datum.yolo_labels(i).y_min();
        const Dtype x_max = datum.yolo_labels(i).x_max();
        const Dtype y_max = datum.yolo_labels(i).y_max();
        const Dtype height = y_max - y_min;
        const Dtype width = x_max - x_min;

        label[i * 5 + 2] = (x_min + x_max) / (2 * width);  //center_x
        label[i * 5 + 3] = (y_min + y_max) / (2 * height); //center_y
        label[i * 5 + 4] = width;                          //width
        label[i * 5 + 5] = height;                         //height
    }
}

INSTANTIATE_CLASS(YoloDataLayer);
REGISTER_LAYER_CLASS(YoloData);

}  // namespace caffe
