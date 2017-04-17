function [estimated_labels, class_mse] = test_pca(test_tdm, class_tdm, K)

    [~, N] = size(test_tdm);
    estimated_labels = zeros(1, N);
    class_mse = cell(K, 1);
    
    lse = inf(1, N);
    
    for i = 1:K
        
        M = class_tdm{i}.M_red;
        PC = class_tdm{i}.pc(:, 1:M);
        
        X = test_tdm - repmat(class_tdm{i}.mu, 1, N);
        
        Y = ((PC' * PC) \ PC') * X;
        
        X_hat = PC * Y;
        
        dist = sqrt(sum((X - X_hat).^2, 1));
        class_mse{i} = dist;
        
        idx = dist < lse;
        lse(idx) = dist(idx);
        estimated_labels(idx) = i;
        
    end
    
end