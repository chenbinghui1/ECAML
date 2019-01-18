#include <algorithm>
#include "stdio.h"
#include "caffe/layers/BIER_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
namespace caffe {

template <typename Dtype>
void BIERLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  F_iter_size_ = 0;
  B_iter_size_ = 0;
  for(int i = 0; i < bottom.size()-1; i++){
        center_temp_.push_back(vector<Dtype>(bottom[i]->channels(),Dtype(0.)));//save temp
  }
}
template <typename Dtype>
void BIERLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* label = bottom[bottom.size()-1]->cpu_data();
    int num = bottom[0]->num();//
    int M = bottom.size()-1;//number of learners
    Dtype beta = this->layer_param_.bier_loss_param().beta();
    Dtype alpha = this->layer_param_.bier_loss_param().alpha();
    Dtype cost_p = this->layer_param_.bier_loss_param().cost_p();
    Dtype cost_n = this->layer_param_.bier_loss_param().cost_n();
    Dtype shrinkage = this->layer_param_.bier_loss_param().shrinkage();
    Dtype loss = 0.;
    top[0]->mutable_cpu_data()[0]=0.;
    for(int i = 0; i < M; i++)
    {
        caffe_set(bottom[i]->count(),Dtype(0.),bottom[i]->mutable_cpu_diff());
    }
    //compute wij for positive number and negative number
    int w_pos=0, w_neg=0;
    for(int i=0; i<num; i++)
        for(int j=i+1; j<num; j++)
            label[i]==label[j] ? w_pos++ : w_neg++;
    
    for(int i=0; i<num; i++)
    {
                for(int j=i+1; j<num; j++)
	        {
	                int w = label[i]==label[j] ? w_pos : w_neg;
	                Dtype w_n = 1.0;
	                std::vector<Dtype> s(M);
	                std::vector<Dtype> eta(M);
	                Dtype param = label[i]==label[j] ? (-1.0*alpha*cost_p) : (alpha*cost_n);
	                for(int m = 0; m < M; m++)
	                {
                                eta[m] = Dtype(2.0/(m+2.0));
                                int channels = bottom[m]->channels();
                                Dtype inner = caffe_cpu_dot(channels, bottom[m]->cpu_data() + i * channels, bottom[m]->cpu_data() + j * channels);
                                Dtype norm_i = std::sqrt(caffe_cpu_dot(channels, bottom[m]->cpu_data() + i * channels, bottom[m]->cpu_data() + i * channels));
                                Dtype norm_j = std::sqrt(caffe_cpu_dot(channels, bottom[m]->cpu_data() + j * channels, bottom[m]->cpu_data() + j * channels));
                                
                                //s[m] = (m==0) ? eta[m]*inner/(norm_i*norm_j) : (1.0-eta[m])*s[m-1]+eta[m]*inner/(norm_i*norm_j);
                                //add new code
                                s[m] = (m==0) ? 0.5+ shrinkage*Dtype(-1.0)*param*(inner/(norm_i*norm_j)-beta) : s[m-1]+shrinkage*Dtype(-1.0)*param*(inner/(norm_i*norm_j)-beta);
                                
                                
		                Dtype ans = exp(param*(inner/(norm_i*norm_j)-beta));
		                //gradients
		                //xi
		                caffe_copy(channels, bottom[m]->cpu_data() + j * channels, &center_temp_[m][0]);
		                caffe_cpu_axpby(channels, Dtype(-1.0*inner/(norm_i*norm_i*norm_i*norm_j)), bottom[m]->cpu_data() + i * channels, Dtype(1.0/(norm_i*norm_j)), &center_temp_[m][0]);
		                caffe_cpu_axpby(channels, Dtype(w_n*param*ans/(Dtype(w)+Dtype(w)*ans)/M), &center_temp_[m][0], Dtype(1), bottom[m]->mutable_cpu_diff() + i * channels);
		                //xj
		                caffe_copy(channels, bottom[m]->cpu_data() + i * channels, &center_temp_[m][0]);
		                caffe_cpu_axpby(channels, Dtype(-1.0*inner/(norm_j*norm_j*norm_j*norm_i)), bottom[m]->cpu_data() + j * channels, Dtype(1.0/(norm_i*norm_j)), &center_temp_[m][0]);
		                caffe_cpu_axpby(channels, Dtype(w_n*param*ans/(Dtype(w)+Dtype(w)*ans)/M), &center_temp_[m][0], Dtype(1), bottom[m]->mutable_cpu_diff() + j * channels);
		                
		                //w_n = exp(param*(s[m]-beta))/((1.+exp(param*(s[m]-beta))));
		                //add new code
		                //w_n=exp(Dtype(-1)*s[m])/(1.0+exp(Dtype(-1)*s[m]));
		                //w_n = (label[i]==label[j])? w_n*Dtype(1) : w_n*Dtype(2);
		                w_n=Dtype(1);
		        }
		        //Dtype ans = exp(param*(s[M-1]-beta));
		        //loss = log(Dtype(1.0+ans)); 
		        //add new codes
		        loss = log(Dtype(1)+exp(Dtype(-1)*s[M-1]));
		        top[0]->mutable_cpu_data()[0] = top[0]->mutable_cpu_data()[0] + loss/w;
	
	        }
	
	
    }
    
}

     

template <typename Dtype>
void BIERLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    //update gradients
    for(int m = 0; m < bottom.size()-1; m++)
    {
        caffe_cpu_scale(bottom[m]->count(),top[0]->cpu_diff()[0],bottom[m]->cpu_diff(),bottom[m]->mutable_cpu_diff());
    }

}




#ifdef CPU_ONLY
STUB_GPU(BIERLossLayer);
#endif

INSTANTIATE_CLASS(BIERLossLayer);
REGISTER_LAYER_CLASS(BIERLoss);

}  // namespace caffe
