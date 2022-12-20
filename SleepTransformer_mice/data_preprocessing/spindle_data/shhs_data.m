clear all
close all
clc

addpath('edf_reader');

xml_path = 'C:/Users/javig/shhs/polysomnography/annotations-events-nsrr/shhs1/';
edf_path = 'C:/Users/javig/shhs/polysomnography/edfs/shhs1/';
mat_path = './mat/';

if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([edf_path, '*.edf']);
N = numel(dirlist);


for n = 1 : N
    filename = dirlist(n).name;
    disp(filename);
    process_and_save_1file(filename, n, xml_path, edf_path, mat_path);
end