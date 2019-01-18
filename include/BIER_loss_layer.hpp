#ifndef CAFFE_BIER_LOSS_LAYER_HPP_
#define CAFFE_BIER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class BIERLossLayer : public LossLayer<Dtype> {
 public:
  explicit BIERLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "BIERLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<vector<Dtype> > center_temp_;//vector<Blob<Dtype>*>
  int F_iter_size_;
  int B_iter_size_;

};
}

#endif
