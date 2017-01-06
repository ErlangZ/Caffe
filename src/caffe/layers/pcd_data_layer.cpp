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

#include "pcd/grid.h"
#include "pcd/types.h"
#include "pcd/label_reader.h"

namespace caffe {

using namespace adu::perception;

template <typename Dtype>
PCDDataLayer<Dtype>::~PCDDataLayer<Dtype>() {
    this->StopInternalThread();
}


template <typename Dtype>
void PCDDataLayer<Dtype>::DataLayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                                         const std::vector<Blob<Dtype>*>& top) {

    _pcd_root = this->layer_param_.pcd_data_param().root_folder();
    CHECK(_label_reader.init(this->layer_param_.pcd_data_param().label_file()));
    for (typename LabelsReader::Iter iter = _label_reader.begin();
         iter != _label_reader.end(); iter++) {
        _file_names.push_back(iter->first);
    }
    
    _batch_size = this->layer_param_.pcd_data_param().batch_size();

    _grid_x_num = this->layer_param_.pcd_data_param().grid_x_num();
    _grid_y_num = this->layer_param_.pcd_data_param().grid_y_num();
    _grid_z_num = this->layer_param_.pcd_data_param().grid_z_num();
    _top_shape.resize(4); 
    _top_shape[0] = _batch_size;
    _top_shape[1] = _grid_x_num;
    _top_shape[2] = _grid_y_num; 
    _top_shape[3] = _grid_z_num;
    CHECK(_grid_x_num > 0 && _grid_y_num > 0 && _grid_z_num > 0) << "Grid Size Should be positive";
}

// This function is called on prefetch thread
// FIXME:Seems the function should be thread-safe.
template <typename Dtype>
void PCDDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    //Reshape Top Blob
    batch->data_.Reshape(_top_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();
    int batch_count = 0;
    while (batch_count < _batch_size) {
        //TODO: Set Random Seed.
        int index = rand() % _file_names.size();
        Label::Ptr label = _label_reader.get(_file_names[index]);

        //Read PCD 
        std::string file_path = _pcd_root + _file_names[index];
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud = read_pcd(file_path);
        //Put Data into Net Top
        for (size_t box_index = 0; box_index < label->boxes.size(); box_index++) {
            const Box::Ptr box = label->boxes[box_index];

            Dtype* data = prefetch_data + batch->data_.offset(batch_count);
            Grid<Dtype> grid(_grid_x_num, _grid_y_num, _grid_z_num, data);
            if (!grid.put_point_cloud_to_grids(box->get_cloud(point_cloud))) {
                LOG(ERROR) << "Put Point Cloud into Grid failed. Box" << box->debug_string();
                continue;
            }

            prefetch_label[batch_count] = box->get_type();
            batch_count ++;
        }

    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch(size:" << _batch_size << "): " 
               << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(PCDDataLayer);
REGISTER_LAYER_CLASS(PCDData);

}
