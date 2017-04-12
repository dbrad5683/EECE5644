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
trainBias = dataset.bias(trainIdx);
testTDM_full = dataset.tdm(:,testIdx);
testMes = dataset.message(testIdx);
testBias = dataset.bias(testIdx);
dimFull = size(trainTDM_full,1);
K = length(dataset.bias_labels);
%% remove words that only appear once
trainTDM_red = trainTDM_full(dataset.wordCounts ~= 1,:);
testTDM_red = testTDM_full(dataset.wordCounts ~= 1,:);
dimRed = size(trainTDM_red,1);
%% train LDA
[w,backgroundMeans,trainMean,T] = train_LDA(trainTDM_red,trainBias,dataset.bias_labels,200);
%% test points
out = test_LDA(testTDM_red,w,backgroundMeans,trainMean,T,K);
%% results
[~,outClassIdx] = max(out);
outLabel = dataset.bias_labels(outClassIdx);
results = strcmpi(outLabel,testBias);
acc = sum(results)/numTest