clear all
close all
clc


rng(10); % for repeatable

% divide subjects into training, evaluation, and test sets for consistency
% between various networks

% Nsub = 5463;
Nsub = 75;

subjects = randperm(Nsub);

test = 0.3;
train = 0.7; % includes validation

test_sub = sort(subjects(1 : round(test*Nsub)));
rest = setdiff(subjects, test_sub);
perm_list = randperm(numel(rest));

val = 0.1;
val = val/train; % calculating the percetnage relative to the train set (divide 10 by 70)

eval_sub = sort(rest(perm_list(1 : round(val*length(perm_list)))));
train_sub = sort(setdiff(rest, eval_sub));

% % 50 subjects as eval set
% train_check_sub = sort(rest(perm_list(101:200)));
% train_sub = sort(rest(perm_list(101:end)));

% save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub','train_check_sub');
save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub');