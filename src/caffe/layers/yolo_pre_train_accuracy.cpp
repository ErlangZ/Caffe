#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/yolo_pre_train_accuracy.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloPreTrainAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                                  const vector<Blob<Dtype>*>& top) {
    //bottom[0] is label
    //bottom[1...C] is output
    error_ = Dtype(0.01);
}

template <typename Dtype>
void YoloPreTrainAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                               const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4, 1); 
  top_shape[0] = bottom[0]->channels() + 1;
  top[0]->Reshape(top_shape);
  //Top[0] is the all rightness.
  //Top[1...channels] are the rightness of each class.
   
  channels_ = bottom.size()-1;
  for (int i = 1; i <= channels_; i++) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num())<< "Bottom[" << i << "]'s shape is" << bottom[i]->shape_string();
    CHECK_EQ(1, bottom[i]->channels()) << "Bottom[" << i << "]'s channels is not 1";
  }
  CHECK_EQ(channels_, bottom[0]->channels());
}

template <typename Dtype>
void YoloPreTrainAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
  int all_right_count = 0;
  vector<int> right(channels_, 0);
  Dtype all(0.0);

  Blob<Dtype>* label_layer = bottom[0];
  const Dtype* label_data = label_layer->cpu_data();

  for(int n = 0; n < label_layer->num(); n++) {
      bool hit = true; 
      for (int c = 1; c <= channels_; c++) {
        const Dtype* output_data = bottom[c]->cpu_data();
        if (output_data[n] > error_ && label_data[c-1] > 1e-6) {
            right[c-1] ++;
        } else if (output_data[n] < error_ && label_data[c-1] <= 1e-6) {
            right[c-1] ++;
        } else {
            hit = false;
        }
      }
      if (hit) all_right_count ++;
      all += 1.0;

      label_data += label_layer->count(1);
  }


//  std::stringstream ss;
  top[0]->mutable_cpu_data()[0] = all_right_count/all;
//  ss << "YoloPreTrainAccuracy all:" << top[0]->cpu_data()[0] * 100 << "% ";
  for (int c = 0; c < right.size(); c++) {
    top[0]->mutable_cpu_data()[c+1] = right[c]/all;
//      ss << " class:" << c-1 << "[" << top[0]->mutable_cpu_data()[c] << "] ";
  }
//  LOG(INFO) << ss.str(); 
// Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(YoloPreTrainAccuracyLayer);
REGISTER_LAYER_CLASS(YoloPreTrainAccuracy);

}  // namespace caffe
