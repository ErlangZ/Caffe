#ifndef CAFFE_UNPOOLING_LAYER_HPP_
#define CAFFE_UNPOOLING_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief UnPools the input image by taking the max, average, etc. within regions.
 * UnPooling use only the Test Phase, no backward function.
 */
template <typename Dtype>
class UNPoolingLayer : public Layer<Dtype> {
 public:
  explicit UNPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UNPooling"; }
  virtual inline int MaxBottomBlobs() const { 
      return (this->layer_param_.pooling_param().pool() == 
              PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- UNPooling cannot be used as when train.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  //int pad_h_, pad_w_; Unpooling should has no padding.
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  //bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

}
#endif 
