#ifndef CAFFE_NPAIR_LOSS_LAYER_HPP_
#define CAFFE_NPAIR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class NpairLossLayer : public LossLayer<Dtype> {
 public:
  explicit NpairLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "NpairLoss"; }
  
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
 

  Blob<Dtype> feature_;  // cached for backward pass
  Blob<Dtype> feature_label_;
  Blob<Dtype> diff_temp_;
  Blob<Dtype> center_temp_;
  Blob<Dtype> center_temp1_;
  Blob<Dtype> center_temp2_;
  Blob<Dtype> diff_;
  Blob<Dtype> loss_temp_;
 
 // Blob<int> flag;
};
}

#endif
