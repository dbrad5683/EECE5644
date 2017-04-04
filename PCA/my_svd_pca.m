function [Y, C, V] = my_svd_pca(X)
% my_svd_pca: Performs PCA using SVD.
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
    
    % Construct the matrix Z
    Z = (1/sqrt(N)) * X';
    
    % Perform SVD on Z
    [~, S, C] = svd(Z);
    
    % Extract singular values and square to get variances
    V = diag(S).^2;
    
    % Project original data onto principal components
    Y = C' * X;
    
end