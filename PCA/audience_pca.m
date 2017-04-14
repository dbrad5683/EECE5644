%% Setup
close all;
clear;
clc;

addpath('~/Documents/SPR17/EECE5644/project');
addpath('~/Documents/SPR17/EECE5644/project/PCA');
load('~/Documents/SPR17/EECE5644/project/dataset.mat');

%% Separate test/train
numTrain = 4000;
numTest = dataset.N - numTrain;

[trainIdx, testIdx] = dataset.get_train_idx(numTrain, numTest);

trainTDM_full = dataset.tdm(:, trainIdx);
trainAudience = dataset.audience(trainIdx);

testTDM_full = dataset.tdm(:, testIdx);
testAudience = dataset.audience(testIdx);

dimFull = size(trainTDM_full, 1);
K = length(dataset.audience_labels);

%% Find words that only appear once
min_freq = 2;
minIdx = dataset.wordCounts > min_freq;

%% Find most common words to ignore
load('../100_most_common_words.mat');
comIdx = zeros(dataset.numWords, 100);
for ii = 1:length(most_common)
    comIdx(:, ii) = strcmpi(most_common{ii}, dataset.wordList);
end
comIdx = sum(comIdx, 2);

%% Remove words
remIdx = minIdx|comIdx;
trainTDM_red = trainTDM_full(remIdx, :);
testTDM_red = testTDM_full(remIdx, :);
dimRed = size(trainTDM_red, 1);

%% Train PCA
min_var_ratio = 1; % Set this in (0:1] to reduce dataset dimensionality
class_tdm = train_pca(trainTDM_red, trainAudience, dataset.audience_labels, min_var_ratio);

%% Test points
estimated_labels = test_pca(testTDM_red, class_tdm, K);

%% Results
outLabel = dataset.audience_labels(estimated_labels);
results = strcmpi(outLabel, testAudience);
acc = sum(results) / numTest