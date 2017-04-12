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
%% find words taht only appear once
min_freq = 2;
minIdx = dataset.wordCounts > min_freq;
%% find most common words to ignore
load('../100_most_common_words.mat');
comIdx = zeros(dataset.numWords,100);
for ii=1:length(most_common)
    comIdx(:,ii) = strcmpi(most_common{ii},dataset.wordList);
end
comIdx = sum(comIdx,2);
%% remove words
remIdx = minIdx|comIdx;
trainTDM_red = trainTDM_full(remIdx,:);
testTDM_red = testTDM_full(remIdx,:);
dimRed = size(trainTDM_red,1);
%% train LDA
[w,backgroundMeans,trainMean,T] = train_LDA(trainTDM_red,trainBias,dataset.bias_labels,2000);
%% test points
out = test_LDA(testTDM_red,w,backgroundMeans,trainMean,T,K);
%% results
[~,outClassIdx] = max(out);
outLabel = dataset.bias_labels(outClassIdx);
results = strcmpi(outLabel,testBias);
acc = sum(results)/numTest