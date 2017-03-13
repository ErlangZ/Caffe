#ifndef CAFFE_YOLO_DATA_LAYER_HPP_
#define CAFFE_YOLO_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"


namespace caffe {

/**
 * @brief Provides data to the Net by assigning tops directly.
 */
template <typename Dtype>
class YoloDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
 YoloDataLayer(const LayerParameter& param);
 ~YoloDataLayer();
 virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 // Data layers should be shared by multiple solvers in parallel
 virtual inline bool ShareInParallel() const { return false; }
 virtual inline const char* type() const { return "YoloData"; }
 virtual inline int ExactNumBottomBlobs() const { return 0; }
 virtual inline int MinTopBlobs() const { return 1; }
 virtual inline int MaxTopBlobs() const { return 2; }
 void init_yolo_label(Dtype* label, const Datum& datum); 
protected:
 virtual void load_batch(Batch<Dtype>* batch);
 DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
