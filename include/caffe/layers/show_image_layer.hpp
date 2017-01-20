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
#include "caffe/util/io.hpp" 
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
  explicit ShowImageLayer(const LayerParameter& param) : Layer<Dtype>(param) {
      cv_data_ = NULL;
  }
  virtual ~ShowImageLayer() {
      delete[] cv_data_;
  };

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      ShowImageParameter show_image_param = this->layer_param_.show_image_param();
      picture_name = show_image_param.image_name(); 
      wait_ = show_image_param.wait_for_key();
      normalize_ = show_image_param.normalize();
      scale_ = show_image_param.scale();
      has_mean_file_ = false;
      if (show_image_param.has_mean_file()) {
          has_mean_file_ = true;
          BlobProto blob_proto;                                                                           
          ReadProtoFromBinaryFileOrDie(show_image_param.mean_file(), &blob_proto);                                   
          mean_data_.FromProto(blob_proto); 
      }
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      num_ = bottom[0]->num();
      channels_ = bottom[0]->channels();
      width_ = bottom[0]->width();
      height_ = bottom[0]->height();
      const int size = num_ * channels_ * height_ * width_;
      cv_data_ = new Dtype[size];

      if (has_mean_file_) {                                                                              
          CHECK_EQ(channels_, mean_data_.channels());                                                
          CHECK_EQ(height_, mean_data_.height());                                                    
          CHECK_EQ(width_, mean_data_.width());                                                      
      }      
  }

  virtual inline const char* type() const { return "ShowImage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  int rbg_channel(int c) {
      switch(c) {
          case 0: return 2;
          case 1: return 1;        
          case 2: return 0;
          default: return -1;
      }
  }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      for (int i = 0; i < propagate_down.size(); ++i) {
        if (propagate_down[i]) { NOT_IMPLEMENTED; }
      }
  }
  
  int num_;
  int channels_;
  int width_;
  int height_;
  bool wait_;
  bool normalize_;

  bool has_mean_file_;
  Blob<Dtype> mean_data_;

  Dtype scale_;
  Dtype* cv_data_;
  std::string picture_name;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
