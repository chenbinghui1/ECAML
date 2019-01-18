## Code for AAAI 2019 Paper "[Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering](bhchen.cn)"

This code is developed based on [Caffe](https://github.com/BVLC/caffe/).

This code also consists the baseline methods used in our paper, such as Triplet loss, Npair loss and Binomial loss.

## Prerequisites
- Caffe
- GPU Memory >= 11G
- Matlab(used for evaluation)

## Training
- Add our files (in include/, src/ and tools/) to your caffe master 
- Merge file caffe.proto into your own caffe.proto
```
Note only that you should replace your code of {message InnerProductParameter{...}} with our provided code
```
- Then complie caffe
```
 cd ~/caffe-master 
 make all -j8
```
- Copy our examples/car/  to your caffe-master, and create folders run/, features/, car_ims_256_nopadding/
```
  cp -r ~/ECAML-master/examples/car ~/caffe-master/examples/
  mkdir ~/caffe-master/examples/car/run
  mkdir ~/caffe-master/examples/car/features
  mkdir ~/caffe-master/examples/car/car_ims_256_nopadding
```
- Download images of [Cars196](http://imagenet.stanford.edu/internal/car196/car_ims.tgz) dataset, and move car_ims/ to ~/caffe-master/examples/car/car_ims_256_nopadding
- Download our [training_list]() (~200MB), and move it to ~/caffe-master/examples/car/
```
you can create your own list, as in our paper, the list is created by random selecting in 65*2 manner (with 65 classes and 2 images per class)
```
- Download [googlenet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) model to ~/caffe-master/examples/car/
- Then train the model by ./finetuen.sh
```
uncomment "energy confusion" codes in train_confusion.prototxt to use our method
```
## Extract features
- After training, the model will be stored at folder run/, then run ./extractfeatures.sh to extract testing image features(feature files will be stored in folder features/)

## Evaluation
- run code in folder ~/ECAML-master/evaluation 

## Citation
If our code is helpful, please kindly cite the following papers:
```
@inproceedings{songCVPR16,
    Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
    Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    Year = {2016}
}
@InProceedings{chen2019energy,
author = {Chen, Binghui and Deng, Weihong},
title = {Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering},
booktitle = {AAAI Conference on Artificial Intelligence},
year = {2019}
}
```
