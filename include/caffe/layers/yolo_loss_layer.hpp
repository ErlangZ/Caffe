#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class YoloBox {
public:
    YoloBox(const Dtype* data, bool first_element_is_type) : 
        type(first_element_is_type ? static_cast<int>(data[0]) : -1), //type == -1 means UNKNOWN
        confidence(first_element_is_type ? 1.0: data[0]),
        center_x(data[1]),
        center_y(data[2]),
        width(data[3]),
        height(data[4]) {
        /*
        for (int i = 0; i < 5; i++) {
            CHECK_TRUE(data[i] <= 1.0);
            CHECK_TRUE(data[i] >= 0.0);
        }
        */
    }
    YoloBox(const Dtype x_min, const Dtype y_min, const Dtype _width, const Dtype _height) {
        confidence = 0.0;
        type = -1;
        center_x = x_min + (width / 2.0);
        center_y = y_min + (height / 2.0);
        width = _width;
        height = _height;
    }

    Dtype min_x() const {
        return center_x - width / 2.0;
    }
    Dtype max_x() const {
        return center_x + width / 2.0;
    }
    Dtype min_y() const {
        return center_y - height / 2.0;
    }
    Dtype max_y() const {
        return center_y + height / 2.0; 
    }
    Dtype area() const {
        return height * width;
    }
    Dtype compute_iou(const YoloBox<Dtype>& other) {
        const Dtype iou_min_x = std::max(min_x(), other.min_x());
        const Dtype iou_max_x = std::min(max_x(), other.max_x());
        if (iou_min_x >= iou_max_x) {
            return 0.0;
        }

        const Dtype iou_min_y = std::max(min_y(), other.min_y());
        const Dtype iou_max_y = std::min(max_y(), other.max_y());
        if (iou_min_y >= iou_max_y) {
            return 0.0;
        }
        return (iou_max_x - iou_min_x) * (iou_max_y - iou_min_y) / area();
    }

    void fill_coord_mem(Dtype* data) {
        data[0] = 1.0;
        data[1] = center_x;
        data[2] = center_y;
        data[3] = width;
        data[4] = height;
    }

    void fill_type_mem(Dtype* data) {
        memset(data, 0, sizeof(Dtype) * 20);
        CHECK(type >= 0);
        data[type] = 1.0;
    }
public:
    int type;
    Dtype confidence;
    Dtype center_x; 
    Dtype center_y;
    Dtype width;
    Dtype height;
};

template <typename Dtype>
void build_boxes_from_labels(const Dtype* labels, std::vector<YoloBox<Dtype> >* labels_boxes) {
    const int boxes_number = static_cast<int>(labels[0]);
    const bool first_element_is_type = true;
    for (int i = 0; i < boxes_number; i++) {
        labels_boxes->push_back(YoloBox<Dtype>(labels + 1 + i * 5, first_element_is_type));
    }
}

//data 0-5 are the first box. 
//data 6-10 are the second box. 
//data 11-30 are the class value. Sigmoid_Output
template <typename Dtype>
void build_boxes_from_output(const Dtype* data, std::vector<YoloBox<Dtype> >* output_box) {
    const bool first_element_is_type = false;
    output_box->push_back(YoloBox<Dtype>(data, first_element_is_type));
    output_box->push_back(YoloBox<Dtype>(data + 5, first_element_is_type));
}

template <typename Dtype>
void fill_the_type(const YoloBox<Dtype>& box, std::vector<int> types) {
    const Dtype range = 1.0 / S_;
    for (int i = 0; i < S_; i++) {
        for (int j = 0; j < S_; j++) {
            YoloBox<Dtype> grid(i * range, j * range, range, range);
            if (grid.compute_iou(box) > 0.4) {
                types[i * S_ + j] = box.type;
            }
        }
    }
}

template <typename Dtype>
Dtype square(const Dtype x) {
    return x * x;
}

//Bottom has 7 * 7 + 1 layers
//Top has 1 layer
template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {
 public:
  explicit YoloLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {
      lambda_coord_ = 5.0;
      lambda_noobj_ = 0.5;
      S_ = 7;
      B_ = 2;
      class_number_ = 20;
      count_ = B_ * 5 + class_number_; // (Confidence, Center_x, Center_y, Width, Height) * 2 + 20 Classes
      volume_ = S_ * S_;
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    diff_.ReshapeLike(*bottom[0])
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
  }

  virtual inline const char* type() const { return "YoloLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
 protected:
  int get_offset(const int i, const int j) const {
      return (i * S_ + j) * count_;
  }
 protected:
  /// @copydoc YoloLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  Blob<Dtype> diff_;
  Dtype lambda_coord_;
  Dtype lambda_noobj_;
  int S_;
  int B_;
  int class_number_;
  int volume_;
  int count_; 
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
