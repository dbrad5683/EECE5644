function estimated_labels = test_pca(test_tdm, class_tdm, K)

    [~, N] = size(test_tdm);
    estimated_labels = zeros(1, N);
    
    mu = mean(test_tdm, 2);
    test_tdm_centered = test_tdm - repmat(mu, 1, N);
    
    lse = inf(1, N);
    
    for i = 1:K
        
        Y = class_tdm{i}.pc' * test_tdm_centered(1:class_tdm{i}.M_red, :);
        dist = sqrt(sum(Y.^2, 1));
        
        idx = dist < lse;
        lse(idx) = dist(idx);
        estimated_labels(idx) = i;
        
    end
    
end