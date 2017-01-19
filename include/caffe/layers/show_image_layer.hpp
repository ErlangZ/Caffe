#ifndef CAFFE_SHOW_IMAGE_LAYER_HPP_
#define CAFFE_SHOW_IMAGE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <opencv2/opencv.hpp>
namespace caffe {

/**
 * @brief Show image on the screen 
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ShowImageLayer : public Layer<Dtype> {
 public:
  explicit ShowImageLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual ~ShowImageLayer() {};

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      ShowImageParameter show_image_param = this->layer_param_.show_image_param();
      picture_name = show_image_param.image_name(); 
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      channels_ = bottom[0]->channels();
      width_ = bottom[0]->width();
      height_ = bottom[0]->height();
      cv_data_.Reshape(1, channels_, height_, width_);
  }

  virtual inline const char* type() const { return "ShowImage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      for (int i = 0; i < propagate_down.size(); ++i) {
        if (propagate_down[i]) { NOT_IMPLEMENTED; }
      }
  }
  
  int channels_;
  int width_;
  int height_;
  Blob<Dtype> cv_data_;
  std::string picture_name;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
