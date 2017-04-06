clear; clc; close all
%%
dataset = post_data('political_social_media_mod.csv');
%%
dataset.calc_tdm;
%%
policy_tdm = dataset.tdm_by_message('policy');
%%
[trainIdx,testIdx] = dataset.get_train_idx(4000,1000);