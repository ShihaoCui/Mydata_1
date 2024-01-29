clear all;
close all;
clc;

load('x_train.mat')
load('y_test.mat')
load('y_train.mat')
load('x_test.mat')
load('x_test_no.mat')
load('y_test_no.mat')
load('x_test1.mat')
load('y_test1.mat')

load('x_selected.mat')

y_train1 = [y_train;y_test_no];
x_train1 = [x_train;x_test_no];

x_test1 = x_test1;
y_test1 = y_test1;





