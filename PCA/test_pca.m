function estimated_labels = test_pca(test_tdm, class_tdm, K)

    [~, N] = size(test_tdm);
    estimated_labels = zeros(1, N);
    
    lse = inf(1, N);
    
    for i = 1:K
        
        M = class_tdm{i}.M_red;
        
        test_tdm_centered = test_tdm(1:M, :) - repmat(class_tdm{i}.mu(1:M), 1, N);
        
        Y = class_tdm{i}.pc' * test_tdm_centered;
        dist = sqrt(sum(Y.^2, 1));
        
        idx = dist < lse;
        lse(idx) = dist(idx);
        estimated_labels(idx) = i;
        
    end
    
end