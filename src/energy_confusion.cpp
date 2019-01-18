#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/energy_confusion.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
namespace caffe {

template <typename Dtype>
void EnergyConfusionLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  center_temp_.Reshape(1,bottom[0]->channels(), 1, 1); //
  diff_.Reshape(bottom[0]->num(),bottom[0]->channels(), 1, 1);
  flag_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
}
template <typename Dtype>
void EnergyConfusionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();//
    int channels = bottom[0]->channels();
    //for random shuffle
    int random_num = this->layer_param_.energy_confusion_loss_param().random_num();
    vector<int> place(bottom[0]->num());
    //random end
    Dtype loss = 0;
    Dtype* diff = diff_.mutable_cpu_data();
    Dtype* center_temp = center_temp_.mutable_cpu_data();
    caffe_set(channels*num, Dtype(0), diff);
    top[0]->mutable_cpu_data()[0]=0;
    //random selected place flags
    caffe_set(flag_.count(), 0, flag_.mutable_cpu_data());
    //compute wij for positive number and negative number
    int w_pos=0, w_neg=0;
    for(int i=0; i<num; i++){
               if(i<num-random_num*2){
                        for(int k = 0; k < bottom[0]->num(); k++)//reset place
                                place[k]=k;
                        std::random_shuffle(place.begin()+i+2, place.end());//random seed
                        for(int j = 0; j<random_num; j++){
                                flag_.mutable_cpu_data()[i*num+place[i+2+j]] = 1;
                        }
                }

                for(int j=i+1; j<num; j++)
                        label[i]==label[j] ? w_pos++ : w_neg++;
        }
    
    for(int i=0; i<num; i++)
    {
        Dtype norm_i = std::sqrt(caffe_cpu_dot(channels, data + i * channels, data + i * channels));
        for(int j=i+1; j<num; j++)
	{
	        if(flag_.cpu_data()[i*num + j]==1){
                                int w = label[i]==label[j] ? w_pos : w_neg;
                                
		                //for euclidean loss
		                caffe_sub(channels, data + i * channels, data + j * channels, center_temp);
		                loss = caffe_cpu_dot(channels, center_temp, center_temp) * 0.5;
		                top[0]->mutable_cpu_data()[0] = top[0]->cpu_data()[0] + loss/w;
		                
		                //gradients
		                caffe_cpu_axpby(channels, Dtype(1)/w, center_temp, Dtype(1), diff + i * channels);
		                caffe_cpu_axpby(channels, Dtype(-1)/w, center_temp, Dtype(1), diff + j * channels);
	        }
	
	}
    }
}

     

template <typename Dtype>
void EnergyConfusionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    //update gradients
    // divided by top[0]->cpu_data()[0] is for log(E)
    caffe_cpu_axpby(bottom[0]->count(), top[0]->cpu_diff()[0]/top[0]->cpu_data()[0], diff_.cpu_data(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
	
}




//#ifdef CPU_ONLY
//STUB_GPU(kcenterLossLayer);
//#endif

INSTANTIATE_CLASS(EnergyConfusionLossLayer);
REGISTER_LAYER_CLASS(EnergyConfusionLoss);

}  // namespace caffe




