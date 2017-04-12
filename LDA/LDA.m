clear; clc; close all;
%% load data class
addpath('~/Documents/Northeastern/2017S/EECE5644/project/EECE5644');
load('~/Documents/Northeastern/2017S/EECE5644/project/EECE5644/dataset.mat');
%% separate test/train
numTrain = 4000;
numTest = dataset.N-numTrain;
[trainIdx,testIdx] = dataset.get_train_idx(numTrain,numTest);
trainTDM_full = dataset.tdm(:,trainIdx);
trainMes = dataset.message(trainIdx);
testTDM_full = dataset.tdm(:,testIdx);
testMes = dataset.message(testIdx);
dimFull = size(trainTDM_full,1);
K = length(dataset.message_labels);
%% train LDA
[w,backgroundMeans,trainMean,T] = train_LDA(trainTDM_full,trainMes,dataset.message_labels,2000);
%% test points
out = test_LDA(testTDM_full,w,backgroundMeans,trainMean,T,K);
%% results
[~,outClassIdx] = max(out);
outMes = dataset.message_labels(outClassIdx);
results = strcmpi(outMes,testMes);
acc = sum(results)/numTest