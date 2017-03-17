// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/03/17 15:22:40
// @file src/caffe/layers/yolo_accuracy_layer.cpp
// @brief 
// 
#include <vector>

#include "caffe/layers/yolo_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void YoloAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                       const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* output_blob = bottom[0];
    Blob<Dtype>* label_blob = bottom[1];
    Dtype map = 0.0; 
    for (int n = 0; n < label_blob->num(); n++) {
        //build boxes from labels.
        std::vector<YoloBox<Dtype> > labels_boxes;
        const Dtype* label_data = label_blob->cpu_data() + label_blob->offset(n);
        build_boxes_from_labels(label_data, &labels_boxes);
 
        std::vector<YoloBox<Dtype> > output_boxes;
        for (int i = 0; i < S_; i++) {
        for (int j = 0; j < S_; j++) {
            const int offset = get_offset(i, j);
            const Dtype* output_data = output_blob->cpu_data() + output_blob->offset(n) + offset;
            build_boxes_from_output(output_data, &output_boxes);
            for (int k = 0; k < output_boxes.size(); k++) {
                if (output_boxes[k].confidence > this->threshold_) {

                    int class_grid_x = output_boxes[k].center_x * S_;
                    class_grid_x = class_grid_x == S_ ? S_-1 : class_grid_x;
                    int class_grid_y = output_boxes[k].center_y * S_;
                    class_grid_y = class_grid_y == S_ ? S_-1 : class_grid_y;
                    const int class_grid_offset = get_offset(class_grid_x, class_grid_y);
                    const Dtype* class_output = output_blob->cpu_data() + output_blob->offset(n) + class_grid_offset;
                    Dtype poss = -1.0;
                    output_boxes[i].type = -1;
                    for (int c = 0; c < class_number_; c++) {
                        if (class_output[B_ * 5 + c] > poss) {
                            output_boxes[i].type = c;
                            poss = class_output[B_ * 5 + c];
                        }
                    }

                    for (int l = 0; l < labels_boxes.size(); l++) { 
                        if (labels_boxes[l].type == output_boxes[l].type) {
                            Dtype iou = labels_boxes[l].compute_iou(output_boxes[k]); 
                            if (iou > this->threshold_) {
                                map += iou;
                            }
                        }
                    }
                }
            }
        }
        }
    }
    map /= label_blob->num();
    top[0]->mutable_cpu_data()[0] = map;
} 
#ifdef CPU_ONLY
STUB_GPU(YoloAccuracyLayer);
#endif
INSTANTIATE_CLASS(YoloAccuracyLayer);
REGISTER_LAYER_CLASS(YoloAccuracy);

}
