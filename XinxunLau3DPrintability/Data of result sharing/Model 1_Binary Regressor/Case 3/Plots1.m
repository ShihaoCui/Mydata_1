clear all;
clc;
close all;

load('y_train_predicted.mat')
load('y_test_predicted.mat')
load('y_test.mat')
load('y_train.mat')
load('loss.mat')

figure
y_test_predicted1=sign(y_test_predicted-0.5)==1;
y_test_predicted1 = double(y_test_predicted1');
plot(y_test_predicted1)
hold on 
y_test = y_test';
plot(y_test)

figure
y_train_predicted1=sign(y_train_predicted-0.5)==1;
y_train_predicted1 = double(y_train_predicted1);
plot(y_train_predicted1)
hold on 
y_train = y_train';
plot(y_train)

figure
plot(loss)
for i=1:1:100
    ID(i,1) = i*10;
    LOSS(i,1) = loss(i*10);
end

figure
C = confusionmat(y_train, y_train_predicted1);
confusionchart(C);

figure
err_train = y_train-y_train_predicted1;
plot(err_train)

figure
C1 = confusionmat(y_test,y_test_predicted1);
confusionchart(C1);