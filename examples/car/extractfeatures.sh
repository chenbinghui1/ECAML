#!/usr/bin/env sh
EX_PATH=/home/me/caffe/examples/car
MODEL_PATH=$EX_PATH"/run"
FEATURE_PATH=$EX_PATH"/features"

for j in $(seq 1)
do
i=`expr \( $j + 1 \) \* 1000`
echo $i;
modelname=$MODEL_PATH/model_iter_$i.caffemodel
despath=$FEATURE_PATH/model_512_$i
echo $modelname;
#.fea 不用加 .cpp里面加了
/home/me/caffe/build/tools/extract_features \
     $modelname \
     "fea_extract.prototxt" \
     "fc_embedding" \
     $despath \
     41 \
     GPU \
     0
done

