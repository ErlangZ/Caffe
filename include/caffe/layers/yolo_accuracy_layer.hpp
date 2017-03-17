#ifndef CAFFE_YOLO_ACCURACY_LAYER_HPP_
#define CAFFE_YOLO_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/yolo_loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for Yolo Detection task.
 */
template <typename Dtype>
class YoloAccuracyLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit YoloAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
      threshold_ = 0.6;
      S_ = 7;
      B_ = 2;
      class_number_ = 20;
      count_ = B_ * 5 + class_number_;       
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  };
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      top[0]->Reshape(vector<int>());
  }

  virtual inline const char* type() const { return "YoloAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  int get_offset(const int i, const int j) const {
      CHECK(i < S_) << "Offset i:" << i << " j:" << j;
      CHECK(j < S_) << "Offset i:" << i << " j:" << j;
      return (i * S_ + j) * count_;
  }


  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  Dtype threshold_; 
  int S_;
  int B_;
  int class_number_;
  int count_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_ACCURACY_LAYER_HPP_
