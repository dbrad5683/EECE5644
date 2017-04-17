function class_tdm = train_pca(train_tdm, train_labels, labels, min_var_ratio)

    K = length(labels);
    [M, ~] = size(train_tdm);
    
    class_tdm = cell(K, 1);
    
    for i = 1:K
        
        label = labels{i};
        idx = strcmp(label, train_labels);
        class_tdm{i}.class_total = sum(idx);
        
        [proj, pc, var, mu] = my_svd_pca(train_tdm(:, idx));

        if min_var_ratio < 1
            
            var_cumulative = cumsum(var);        
            var_total = repmat(sum(var), length(var_cumulative), 1);
            var_ratio = var_cumulative ./ var_total;
            M_red = find(var_ratio >= min_var_ratio, 1);
            
        else
            
            M_red = M;
            
        end
        
        class_tdm{i}.proj = proj;
        class_tdm{i}.pc = pc;
        class_tdm{i}.mu = mu;
        class_tdm{i}.M_red = M_red;
        
    end

end