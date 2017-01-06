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
    adu::perception::LabelsReader _label_reader;
};

}
#endif

