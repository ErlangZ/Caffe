#ifndef CAFFE_YOLO_PRETRAIN_LOSS_HPP_
#define CAFFE_YOLO_PRETRAIN_LOSS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief LossLayer for Yolo PreTrainLoss
 *        C + 1 bottom layers; 1 top layer
 *        the first one bottom layer is [N * C * 1 * ] Vaules 0 ~ 1
 *        the next C bottom layers [each are N * 1 * 1 * 1] Value 0~1
 *        1 top layer [1 * 1 * 1 * 1]  
 *
 *  YoloPretrainedLossLayer use Cross-Entropy Loss Function.
 *        -\sum_i [y_i ln(h_i) + (1-y_i)ln(1-h_i)], y \in {0, 1}
 */
template <typename Dtype>
class YoloPretrainedLossLayer : public Layer<Dtype> {
 public:
  explicit YoloPretrainedLossLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "YoloPretrainedLoss"; }
  /**
   * Unlike most loss layers, in the YoloPretrainedLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc YoloPretrainedLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  // CAFFE_YOLO_PRETRAIN_LOSS_HPP_
