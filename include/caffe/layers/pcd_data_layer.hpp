#ifndef CAFFE_PCD_DATA_LAYER_HPP_
#define CAFFE_PCD_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "pcd/label_reader.h"

namespace caffe {
/*
* @brief Provides data to the Net from pcd files.
*/
template<typename Dtype> 
class PCDDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit PCDDataLayer(const LayerParameter& param): 
        BasePrefetchingDataLayer<Dtype>(param) {
    }
    virtual ~PCDDataLayer();
    virtual inline const char* type() const { return "PCDData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }
protected:
    virtual void DataLayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                                const std::vector<Blob<Dtype>*>& top);
    virtual void load_batch(Batch<Dtype>* batch);
private:
    void put_data_label(const int index, 
                        const adu::perception::Box::Ptr box,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                        Dtype* prefetch_data,
                        Dtype* prefetch_label);
private:
    adu::perception::LabelsReader _label_reader;
    std::vector<std::string>      _file_names;
    std::string                   _pcd_root;
    int _batch_size;
    int _grid_x_num;
    int _grid_y_num;
    int _grid_z_num;
    std::vector<int> _top_shape;
};

}
#endif

