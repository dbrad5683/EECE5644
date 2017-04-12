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
%% de-mean everything
trainMean = mean(trainTDM_full,2);
trainTDM_dm = trainTDM_full - repmat(trainMean,1,numTrain);
testTDM_dm = testTDM_full - repmat(trainMean,1,numTest);
%% calculate SVD and transform data
disp('SVD');
dim = numTrain;
[U,S,V] = svds(trainTDM_dm,dim);
%% tranform data
T = U*S;
trainTDM = T'*trainTDM_dm;
testTDM = T'*testTDM_dm;
%% visualize
figure; hold on;
idx = cell(K,1);
for ii=1:K
    idx{ii} = strcmpi(dataset.message_labels{ii},dataset.message(trainIdx));
    scatter(trainTDM(1,idx{ii}),trainTDM(2,idx{ii}));
end
%% separate training data into and estimating parameters
disp('estimating parameters');
classTDM = cell(K,1);
classMeans = zeros(dim,K);
classCovs = zeros(dim,dim,K);
classTotal = zeros(K,1);
for ii=1:K
    currMes = dataset.message_labels{ii};
    idx = strcmpi(dataset.message_labels{ii},trainMes);
    classTotal(ii) = sum(idx);
    classTDM{ii} = trainTDM(:,idx);
    % get parameters
    classMeans(:,ii) = mean(classTDM{ii},2);
    classCovs(:,:,ii) = cov(classTDM{ii}');
    disp(['class num: ', num2str(ii)]);
end
classPrior = classTotal./numTrain;
%% get background means and covariances
disp('background parameters');
backgroundMeans = zeros(dim,K);
backgroundCovs = zeros(dim,dim,K);
idx = 1:K;
for ii=1:K
    % make background TDM
    backgroundTDM = trainTDM(:,~strcmpi(dataset.message_labels{ii},dataset.message(trainIdx)));
    backgroundMeans(:,ii) = mean(backgroundTDM,2);
    backgroundCovs(:,:,ii) = cov(backgroundTDM');
    disp(['class num: ', num2str(ii)]);
end
%% get w
disp('calculating w');
w = zeros(dim,K);
for ii=1:K
    w(:,ii) = eye(dim)\(backgroundCovs(:,:,ii)+classCovs(:,:,ii))*(classMeans(:,ii)-backgroundMeans(:,ii));
    disp(['class num: ', num2str(ii)]);
end
%% test points
out = zeros(K,numTest);
for ii=1:K
    % de-mean for background
    test_dm = testTDM - repmat(backgroundMeans(:,ii),1,numTest);
    % project onto w
    out(ii,:) = w(:,ii)'*test_dm;
end
%% results
[~,outClassIdx] = max(out);
outMes = dataset.message_labels(outClassIdx);
results = strcmpi(outMes,testMes);
acc = sum(results)/numTest