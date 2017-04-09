clear; clc; close all;
%% load data class
addpath('~/Documents/Northeastern/2017S/EECE5644/project/EECE5644');
load('~/Documents/Northeastern/2017S/EECE5644/project/EECE5644/dataset.mat');
%% separate test/train
numTrain = 4000;
numTest = 1000;
[trainIdx,testIdx] = dataset.get_train_idx(numTrain,numTest);
trainTDM = dataset.tdm(:,trainIdx);
trainMes = dataset.message(trainIdx);
testTDM = dataset.tdm(:,testIdx);
testMes = dataset.message(testIdx);
dim = size(trainTDM,1);
%% separate training data into classes
K = length(dataset.message_labels);
classTDM = cell(K,1);
classMeans = zeros(dim,K);
classCovs = cell(K,1);
for ii=1:K
    currMes = dataset.message_labels{ii};
    idx = strcmpi(dataset.message_labels{ii},trainMes);
    classTDM{ii} = trainTDM(:,idx);
    % get parameters
    classMeans(:,ii) = mean(classTDM{ii},2);
    classCovs{ii} = cov(classTDM{ii}');
end
