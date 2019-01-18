function evaluate_clustering()

K = 98;   %cub 100 cars 98 ebay 11316 flower51
label_num=8131;%cub 5924  cars 8131  ebay 60502 flower4696
p=importdata('../examples/car/car_test.txt');%cub ../../caffe/examples/model/cub/val.txt
%cars  ../../caffe/examples/model/cars/car_test.txt
%ebay ../../caffe/examples/model/ebay/ebay_test.txt
%flower flower_test.txt

fea1=sprintf('../examples/car/features/model_512_8000.fea');
feafile=fopen(fea1,'rb');
[dims]=fread(feafile,1,'int');
[num]=fread(feafile,1,'int');
feature=fread(feafile,dims*num,'float');
fclose(feafile);
features=reshape(feature,dims,num);
     
features=features./repmat(sqrt(sum(features.^2)),dims,1);
     
     
     
% evaluation
     X=features(:,1:label_num);
     class_ids=p.data+1;
    % kmeans clustering
    fprintf('2d kmeans %d\n', K);
    opts = struct('maxiters', 1000, 'mindelta', eps, 'verbose', 1);
    [center, sse] = vgg_kmeans(X, K, opts);
    [idx_kmeans, d] = vgg_nearest_neighbour(X, center);

    % construct idx
    num = size(X, 2);
    idx = zeros(num, 1);
    for i = 1:K
        index = find(idx_kmeans == i);
        [~, ind] = min(d(index));
        cid = index(ind);
        idx(index) = cid;
    end

    fprintf('Number of clusters: %d\n', length(unique(idx)));
    save(sprintf('idx_kmeans_googlenet_m128_embed%d_e.mat',dims), 'idx');
% end

% evaluation
num_validation_classes = K;

% load ground truth from filenames
% [image_ids, class_ids, superclass_ids, path_list] = ...
%     textread('/cvgl/group/Ebay_Dataset/Ebay_test.txt', '%d %d %d %s',...
%     'headerlines', 1);
% class_ids=object.label-99;
num = numel(class_ids);
item_ids = cell(num,1);
for i = 1:num
    class_id = class_ids(i);                             
    item_ids{i} = num2str(class_id);
end 
assert(length(unique(item_ids)) == num_validation_classes);

% Given cluster assignment and the class names
%   Compute the three clustering metrics.
[NMI, RI, F1] = compute_clutering_metric(idx, item_ids);

fprintf('[method: m128, distance: e] NMI: %.3f, RI: %.3f, F1: %.3f\n\n', ...
     NMI, RI, F1);
