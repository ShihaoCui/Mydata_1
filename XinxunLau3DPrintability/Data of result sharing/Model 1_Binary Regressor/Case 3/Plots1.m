clear all;
clc;
close all;

load('y_train_predicted.mat')
load('y_test_predicted.mat')
load('y_test.mat')
load('y_train.mat')

figure
y_test_predicted1=sign(y_test_predicted-0.5)==1;
plot(y_test_predicted1)
hold on 
plot(y_test)

figure
y_train_predicted1=sign(y_train_predicted-0.5)==1;
plot(y_train_predicted1)
hold on 
plot(y_train)



