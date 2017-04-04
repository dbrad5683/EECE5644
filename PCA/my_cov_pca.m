function [Y, C, V] = my_cov_pca(X)
% my_cov_pca: Performs PCA using the covariance matrix of X.
%
%   X - MxN matrix of observed data (M variables, N observations)
%
%   Y - MxN matrix of observed data projected onto its principal components
%
%   C - MxM matrix of principal components in terms of observed variables
%       where each column is a principal component in order of descending
%       variance
%
%   V = Mx1 vector of variances
%

    % Determine the size of the data
    [~, N] = size(X);
    
    % Calculate the sample mean of the data (mean of each row)
    mu = mean(X, 2);
    
    % Subtract the mean from each observation
    X = X - repmat(mu, 1, N);
    
    % Calculate the covariance matrix
    K_X = (1/N) * (X * X');
    
    % Calculate the eigenvalues (variances) and eigenvectors (principal
    % components) of the covariance matrix
    [C, V] = eig(K_X, 'vector');
    
    % Sort the variances in descending order and note their indices
    [V, idx] = sort(V, 1, 'descend');
    
    % Rearrange principal components according to sorted variances
    C = C(:, idx);
    
    % Project original data onto principal components
    Y = C' * X;
    
end