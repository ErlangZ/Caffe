#ifndef CAFFE_MAX_ACTIVITY_LAYER_HPP_
#define CAFFE_MAX_ACTIVITY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/* 
 *  Keep the Max Activity(bottom) while set the others Zero.
 */
template <typename Dtype> 
class MaxActivityLayer : public Layer<Dtype> {
public:
    explicit MaxActivityLayer(const LayerParameter& param) : Layer<Dtype>(param) {
        MaxActivityParameter max_activity_param = this->layer_param_.max_activity_param();
        top_ = max_activity_param.top_activity();
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        num_ = bottom[0]->num();
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
    }
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        top[0]->ReshapeLike(*bottom[0]);
    }
    virtual inline const char* type() const { return "MaxActivity"; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
 private:
  int num_;
  int channels_;
  int height_;
  int width_;
  int top_;
};

}

#endif
