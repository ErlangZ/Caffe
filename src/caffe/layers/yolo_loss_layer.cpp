// Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
// @author erlangz(zhengwenchao@baidu.com)
// @date 2017/03/14 15:40:52
// @file src/caffe/layers/yolo_loss_layer.cpp
// @brief 
// 
#include <vector>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                       const vector<Blob<Dtype>*>& top) {
    Dtype loss = 0.0;
    //memset the diff data to zeros.
    memset(diff_.mutable_cpu_data(), 0, sizeof(Dtype) * diff_.count());

    Blob<Dtype>* output_blob = bottom[0];
    Blob<Dtype>* label_blob = bottom[1];
    for (int n = 0; n < bottom[0]->num(); n++) {
        Dtype* diff_data = diff_.mutable_cpu_data() + diff_.offset(n);
        //build boxes from labels.
        std::vector<YoloBox<Dtype> > labels_boxes;
        Dtype label_data = label_blob->cpu_data() + label_blob->offset(n);
        build_boxes_from_labels(label_data, &labels_boxes);
        //build boxes from output.
        std::vector<YoloBox<Dtype> > output_boxes;
        for (int i = 0; i < S_; i++) {
        for (int j = 0; j < S_; j++) {
            const int offset = get_offset(i, j);
            Dtype* output_data = output_blob->cpu_data() + output_blob->offset(n) + offset;
            build_boxes_from_output(output_data, &output_boxes);
        }
        }
        vector<int> contain_object(S_ * S_, -1);  // -1 means No-object.
        vector<bool> responsible(S_ * S_ * B_, false);
        // for each label, find the 'response box' for it.
        for (int i = 0; i < labels_boxes.size(); i++) {
            Dtype best_iou = 0.0;
            int best_iou_index = -1;
            for (int j = 0; j < output_boxes.size(); j++) {
                Dtype iou = labels_boxes[i].compute_iou(output_boxes[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_iou_index = j;
                }
            }

            responsible[j] = true;
            fill_coord_diff(diff_data, best_iou_index);
            loss += lambda_coord_  * 
                    (square(labels_boxes[i].center_x - output_boxes[best_iou_index].center_x) + 
                     square(labels_boxes[i].center_y - output_boxes[best_iou_index].center_y) +
                     square(sqrt(labels_boxes[i].width) - sqrt(output_boxes[best_iou_index].width))+ 
                     square(sqrt(labels_boxes[i].height) - sqrt(output_boxes[best_iou_index].height)));

            fill_confidence_diff(diff_data, best_iou_index); 
            loss += square(1.0 - output_boxes[best_iou_index].confidence);

            fill_the_type(labels_boxes[i], &contain_object);
        }
        // for each no-response-output-box
        for (int i = 0; i < output_boxes.size(); i++) {
            if (!responsible[i]) {
                loss += lambda_noobj_ * square(0.0 - output_boxes[i].confidence);
            }
        }

        // compute the possible.
        Dtype poss[class_number_];
        for (int i = 0; i < contain_object.size(); i++) {
            if (contain_object[i] < 0) {
                continue;
            }
            memset(poss, 0, sizeof(poss));
            poss[contain_object[i]] = 1.0;
            const int offset = i * count_ + B_ * 5; 
            Dtype* output_data = output_blob->cpu_data() + output_blob->offset(n) + offset;
            for (int j = 0; j < class_number_; j++) {
                loss += square(poss[j] - output_data[j]);
            }
        }
    }

    loss /= bottom[0]->num();
    top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, 
                                        const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(YoloLossLayer);
#endif
INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);
}  // namespace caffe
