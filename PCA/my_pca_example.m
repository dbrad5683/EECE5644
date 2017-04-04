close all;
clear;
clc;

set(0, 'DefaultAxesFontsize', 12);

%% Generate random data

% Number of variables
M = 2;

% Number of observations
N = 1000;

% Create N observations of M hidden random variables
X = randn(M, N);

% Map the underlying hidden random variables to recorded observations
%   X = AX + B
A = [2, 1; 1, 1];
B = [-3; 2];
X = (A * X) + repmat(B, 1, N);

% Add some random noise
X = X + (0.1 * randn(M, N));

%% Perform PCA

% Covariance PCA
[Y_cov, C_cov, V_cov] = my_cov_pca(X);

% SVD PCA
[Y_svd, C_svd, V_svd] = my_svd_pca(X);

%% Plot

% Calculate the sample mean of the data (this is done within the PCA 
% functions but do it here again for plotting purposes)
mu = mean(X, 2);
X_zero_mean = X - repmat(mu, 1, N);

figure();
set(gcf, 'papertype', 'usletter')

subplot(5, 2, 1:4);
hold on;
plot(X(1,:), X(2,:), 'k.', 'markersize', 12);
xlabel('X_1');
ylabel('X_2');
title('Original Data');
axis equal;
hold off;

subplot(5, 2, 5);
hold on;
plot(X_zero_mean(1,:), X_zero_mean(2,:), 'k.', 'markersize', 12);
pc1_cov = plot([0, 3*C_cov(1,1)], [0, 3*C_cov(2,1)], 'r-', 'linewidth', 2);
pc2_cov = plot([0, 3*C_cov(1,2)], [0, 3*C_cov(2,2)], 'g-', 'linewidth', 2);
legend([pc1_cov, pc2_cov], {'PC 1', 'PC 2'}, 'Location', 'Southeast')
xlabel('X_1');
ylabel('X_2');
title('Mean-Centered Data with Covariance Principal Components');
axis equal;
hold off;

subplot(5, 2, 6);
hold on;
plot(X_zero_mean(1,:), X_zero_mean(2,:), 'k.', 'markersize', 12);
pc1_svd = plot([0, 3*C_svd(1,1)], [0, 3*C_svd(2,1)], 'r-', 'linewidth', 2);
pc2_svd = plot([0, 3*C_svd(1,2)], [0, 3*C_svd(2,2)], 'g-', 'linewidth', 2);
legend([pc1_svd, pc2_svd], {'PC 1', 'PC 2'}, 'Location', 'Southeast')
xlabel('X_1');
ylabel('X_2');
title('Mean-Centered Data with SVD Principal Components');
axis equal;
hold off;

subplot(5, 2, 7);
hold on;
plot(Y_cov(1,:), Y_cov(2,:), 'k.', 'markersize', 12);
xlabel('PC 1');
ylabel('PC 2');
title('Result of Covariance PCA');
axis equal;
hold off;

subplot(5, 2, 8);
hold on;
plot(Y_svd(1,:), Y_svd(2,:), 'k.', 'markersize', 12);
xlabel('PC 1');
ylabel('PC 2');
title('Result of SVD PCA');
axis equal;
hold off;

subplot(5, 2, 9);
hold on;
plot(Y_cov(1,:), 0, 'k.', 'markersize', 12);
xlabel('PC 1');
title('Covariance PCA with Reduced Dimensionality');
set(gca, 'ytick', [],'ycolor', 'w', 'box', 'off')
axis equal;
hold off;

subplot(5, 2, 10);
hold on;
plot(Y_svd(1,:), 0, 'k.', 'markersize', 12);
xlabel('PC 1');
title('SVD PCA with Reduced Dimensionality');
set(gca, 'ytick', [],'ycolor', 'w', 'box', 'off')
axis equal;
hold off;
