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
trainBias = dataset.bias(trainIdx);

testTDM_full = dataset.tdm(:, testIdx);
testBias = dataset.bias(testIdx);

dimFull = size(trainTDM_full, 1);
K = length(dataset.bias_labels);

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
min_var_ratio = 0.9; % Set this in (0:1] to reduce dataset dimensionality
class_tdm = train_pca(trainTDM_red, trainBias, dataset.bias_labels, min_var_ratio);

%% Test points
[estimated_labels, class_mse] = test_pca(testTDM_red, class_tdm, K);

%% results
outLabel = dataset.bias_labels(estimated_labels);
results = strcmpi(outLabel, testBias);
acc = sum(results) / numTest

bias = zeros(2, numTest);
bias(1, :) = strcmpi(testBias, 'partisan')';
bias(2, :) = strcmpi(testBias, 'neutral')';

label = zeros(2, numTest);
label(1, :) = strcmpi(outLabel, 'partisan')';
label(2, :) = strcmpi(outLabel, 'neutral')';

plotconfusion(bias, label);

figure();
histogram(class_mse{1}(class_mse{1} < 50))
hold on;
histogram(class_mse{2}(class_mse{2} < 50))
legend([dataset.bias_labels(1), dataset.bias_labels(2)])
title('Histogram for Bias Labels')
ylabel('Frequency')
xlabel('Mean Squared Error')
axis square