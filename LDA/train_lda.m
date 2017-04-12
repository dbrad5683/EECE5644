function [w,backgroundMeans,trainMean,T] = train_LDA(trainTDM_full,trainLabels,labels,dim)
    K = length(labels);
    %% de-mean everything
    num = size(trainTDM_full,2);
    trainMean = mean(trainTDM_full,2);
    trainTDM_dm = trainTDM_full - repmat(trainMean,1,num);
    %% calculate SVD and transform data
    disp('SVD');
    [U,S,~] = svds(trainTDM_dm,dim);
    %% tranform data
    T = U;
    trainTDM = T'*trainTDM_dm;
    %% visualize
    figure;
    hold on;
    for ii=1:K
        idx = strcmpi(labels{ii},trainLabels);
        scatter3(trainTDM(1,idx),trainTDM(2,idx),trainTDM(3,idx));
    end
    %% separate training data into and estimating parameters
    disp('estimating parameters');
    classTDM = cell(K,1);
    classMeans = zeros(dim,K);
    classCovs = zeros(dim,dim,K);
    classTotal = zeros(K,1);
    for ii=1:K
        currMes = labels{ii};
        idx = strcmpi(currMes,trainLabels);
        classTotal(ii) = sum(idx);
        classTDM{ii} = trainTDM(:,idx);
        % get parameters
        classMeans(:,ii) = mean(classTDM{ii},2);
        classCovs(:,:,ii) = cov(classTDM{ii}');
        disp(['class num: ', num2str(ii)]);
    end
    classPrior = classTotal./num;
    %% get background means and covariances
    disp('background parameters');
    backgroundMeans = zeros(dim,K);
    backgroundCovs = zeros(dim,dim,K);
    idx = 1:K;
    for ii=1:K
        % make background TDM
        currMes = labels{ii};
        backgroundTDM = trainTDM(:,~strcmpi(currMes,trainLabels));
        backgroundMeans(:,ii) = mean(backgroundTDM,2);
        backgroundCovs(:,:,ii) = cov(backgroundTDM');
        disp(['class num: ', num2str(ii)]);
    end
    %% get w
    disp('calculating w');
    w = zeros(dim,K);
    for ii=1:K
        w(:,ii) = inv(backgroundCovs(:,:,ii)+classCovs(:,:,ii))*(classMeans(:,ii)-backgroundMeans(:,ii));
        disp(['class num: ', num2str(ii)]);
    end
end

