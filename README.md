## Code for AAAI 2019 Paper "[Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering](bhchen.cn)"

This code is developed based on [Caffe](https://github.com/BVLC/caffe/).

This code also consists the baseline methods used in our paper, such as Triplet loss, Npair loss and Binomial loss.

## Prerequisites
- Caffe
- GPU Memory >= 11G
- Matlab(used for evaluation)

## Training
- Add files (in include/ and src/) to your caffe master 
- Merge file caffe.proto into your own caffe.proto
```
Note only that you should replace your code of {message InnerProductParameter{...}} with our provided code
```
- Then complie caffe
```
 cd ~/caffe_master 
 make all -j8
```
- Copy our examples/car/ to your caffe_master, and create some run/ features/ car_ims_256_nopadding/
```
  cp -r ~/ECAML/examples/car ~/caffe_master/examples/
  mkdir ~/caffe_master/examples/car/run
  mkdir ~/caffe_master/examples/car/features
  mkdir ~/caffe_master/examples/car/car_ims_256_nopadding
```
- Download images of [Cars196](http://imagenet.stanford.edu/internal/car196/car_ims.tgz) dataset, and move car_ims/ to ~/caffe_master/examples/car/car_ims_256_nopadding
- Download [training_list](), and move it to ~/caffe_master/examples/car/


