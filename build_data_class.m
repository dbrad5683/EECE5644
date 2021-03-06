clear; clc; close all
%% create dataset class
dataset = post_data('political_social_media_mod.csv');
%% calculate term-document matrix
dataset.calc_tdm;
%% save class instance so you don't have to run this every time
save('dataset.mat','dataset');
