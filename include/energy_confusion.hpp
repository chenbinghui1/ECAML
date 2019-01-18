#ifndef CAFFE_ENERGY_CONFUSION_LOSS_LAYER_HPP_
#define CAFFE_ENERGY_CONFUSION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class EnergyConfusionLossLayer : public LossLayer<Dtype> {
 public:
  explicit EnergyConfusionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "EnergyConfusionLoss"; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> diff_temp_;
  Blob<Dtype> center_temp_;
  Blob<Dtype> diff_;
  Blob<int> flag_;
};
}

#endif
