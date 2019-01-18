#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/layer.hpp"
#include "caffe/layers/Npair.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void NpairLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  diff_temp_.Reshape(1,bottom[0]->channels(), 1, 1); 
  center_temp_.Reshape(1,bottom[0]->channels(), 1, 1); 
  diff_.Reshape(bottom[0]->num(),bottom[0]->channels(), 1, 1);
  center_temp1_.Reshape(1,bottom[0]->channels(),1,1);
  
}

template <typename Dtype>
void NpairLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();//
    int channels = bottom[0]->channels();
    Dtype coeff = this->layer_param_.npair_loss_param().coeff();
    Dtype loss(0.0);
    Dtype k = this->layer_param_.npair_loss_param().k();
	
	Dtype* diff = diff_.mutable_cpu_data();
	
	Dtype* diff_temp = diff_temp_.mutable_cpu_data();
	
	Dtype* center_temp = center_temp_.mutable_cpu_data();
	Dtype* center_temp1 = center_temp1_.mutable_cpu_data();
	
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
	const int kcenter=this->layer_param_.npair_loss_param().kcenter();//
	const int kcenter_n=this->layer_param_.npair_loss_param().kcenter_n();//
	//const float margin=this->layer_param_.kcenterv2_loss_param().margin();//margin
	float distance_temp=0;//distance
	
	int num_flag=0;//
	int num_flag_n=0;//
	
	caffe_set(channels*num, Dtype(0), diff);
	
	top[0]->mutable_cpu_data()[0]=0;
	
	for(int i=0; i<num; i++)
	{
		caffe_set(channels, Dtype(0), center_temp);
		std::vector<std::pair<float, long int> > distance;//
		std::vector<std::pair<float, long int> > distance_n;//
		num_flag=0; 
		//distance.push_back(std::make_pair(Dtype(0), i));
		for (int j=0; j<num; j++)
		{
			
			if(i!=j && label[i]==label[j])
			{
				num_flag++;
				
				caffe_sub(
							channels,
							data+i*channels,  // xi
							data+j*channels,  // xj
							diff_temp);  // xi - xj;
				distance_temp=caffe_cpu_dot(channels,diff_temp,diff_temp);//
				
			    distance.push_back(std::make_pair(std::sqrt(distance_temp), j));
				
			}
			
		}
	//	printf("num_flag=%d\n",num_flag);
		if(num_flag>=kcenter)
		{
			std::partial_sort(distance.begin(), distance.begin() + kcenter,distance.end());//
			for(int j=0;j<kcenter;j++)
			{
				caffe_cpu_axpby(channels,Dtype(1)/kcenter,data+(distance[j].second)*channels,Dtype(1),center_temp);//
			}
			
			num_flag_n=0;
			for (int j=0; j<num; j++)  //
			{
				if(label[i]!=label[j])
				{
					num_flag_n++;
					//std::vector<std::pair<float, long int> > distance;//
					caffe_sub(
								channels,
								data+j*channels,  // xj
								center_temp,  // c
								diff_temp);  // xj - c;
					distance_temp=caffe_cpu_dot(channels,diff_temp,diff_temp);//
					//printf("distance_temp=%f\n",std::sqrt(distance_temp));
					distance_n.push_back(std::make_pair(std::sqrt(distance_temp), j));
				
				}
			 
			}
		  
			if(num_flag_n>=kcenter_n)
			{
				std::partial_sort(distance_n.begin(), distance_n.begin() + kcenter_n,distance_n.end());//
				Dtype x_norm = std::sqrt(caffe_cpu_dot(channels,data+i*channels,data+i*channels));//xi norm
				Dtype center_norm = std::sqrt(caffe_cpu_dot(channels,center_temp,center_temp));//center norm
				Dtype distance_temp1 = caffe_cpu_dot(channels, data + i * channels, center_temp) * (Dtype(2)-Dtype(k)) + (Dtype(1)-Dtype(k))*x_norm*center_norm;;//
				//
				Dtype max1=distance_temp1;
				for(int j=0;j<kcenter_n;j++){
				        Dtype temp_max =caffe_cpu_dot(channels, data + distance_n[j].second * channels, center_temp);
				        max1 = max1>=temp_max ? max1: temp_max;
				}
				distance_temp1 = exp(distance_temp1 - max1);//
				distance_temp=0;//
				for(int j=0;j<kcenter_n;j++)
				{
					
					distance_n[j].first = exp(caffe_cpu_dot(channels, data + distance_n[j].second * channels, center_temp) - max1);
					
					distance_temp+=distance_n[j].first;
				}
				//printf("%12f\n",distance_temp);


				loss = -log(std::max(Dtype(distance_temp1/(distance_temp1 + distance_temp)),Dtype(1e-20))) + caffe_cpu_dot(channels,data + i * channels,data + i * channels)/Dtype(2)*coeff;
				
				if(loss>=0)
				{       //Dtype dis_center_norm = std::sqrt(caffe_cpu_dot(channels,diff_temp,diff_temp));
				        caffe_copy(channels, center_temp, center_temp1);
				        caffe_cpu_axpby(channels, (Dtype(1)-k)*center_norm/x_norm, data + i * channels, Dtype(2)-k, center_temp1);
					caffe_cpu_axpby(channels, (distance_temp1/(distance_temp1 + distance_temp) - Dtype(1)), center_temp1, Dtype(1), diff+i*channels);//
					caffe_cpu_axpby(channels, coeff, data + i * channels, Dtype(1), diff + i * channels);
					//gradients for center
					caffe_copy(channels, data + i * channels, center_temp1);
					caffe_cpu_axpby(channels, (Dtype(1)-k)*x_norm/center_norm, center_temp, Dtype(2)-k, center_temp1);
					caffe_cpu_axpby(channels, distance_temp1/(distance_temp + distance_temp1), center_temp1, Dtype(0), diff_temp);
					for (int j = 0; j < kcenter_n; j++){
					
					         caffe_cpu_axpby(channels, distance_n[j].first/(distance_temp + distance_temp1), data + distance_n[j].second * channels, Dtype(1), diff_temp);
					
					}
					caffe_sub(channels, center_temp1, diff_temp, diff_temp);
					for (int j=0;j<kcenter;j++)
					{
						caffe_cpu_axpby(channels,Dtype(-1)/kcenter,diff_temp,Dtype(1),diff+distance[j].second*channels);//gradients for current data
					}
						
					
					//
					
					for (int j=0;j<kcenter_n;j++)//
					{ 
					
						caffe_cpu_axpby(channels, distance_n[j].first/(distance_temp + distance_temp1), center_temp, Dtype(1), diff+distance_n[j].second*channels);//
						
					
					}
				}
			}
			else
			{
				loss=0;
			}
		}
		else
		{
			loss=0;
		}
		top[0]->mutable_cpu_data()[0] += loss;
	}
	top[0]->mutable_cpu_data()[0] /=num;
	
  
}

     

template <typename Dtype>
void NpairLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
		caffe_cpu_axpby(bottom[0]->count(), top[0]->cpu_diff()[0]/bottom[0]->num(), diff_.cpu_data(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
		
}




#ifdef CPU_ONLY
STUB_GPU(NpairLossLayer);
#endif

INSTANTIATE_CLASS(NpairLossLayer);
REGISTER_LAYER_CLASS(NpairLoss);

}  // namespace caffe
