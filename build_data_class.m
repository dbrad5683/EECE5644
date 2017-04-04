clear; clc; close all
%%
dataset = post_data('political_social_media_mod.csv');
%%
dataset.calc_tdm;
%%
policy_tdm = dataset.tdm_by_label('policy');