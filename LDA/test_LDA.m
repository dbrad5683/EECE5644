function [out] = test_LDA(testTDM_full,w,backgroundMeans,trainMean,T,K)
    num = size(testTDM_full,2);
    out = zeros(K,num);
    %% de mean and transform
    testTDM_dm = testTDM_full - repmat(trainMean,1,num);
    testTDM = T'*testTDM_dm;
    for ii=1:K
        % de-mean for background
        test_dm = testTDM - repmat(backgroundMeans(:,ii),1,num);
        % project onto w
        out(ii,:) = w(:,ii)'*test_dm;
    end
end

