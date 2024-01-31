clear all;
clc;
close all;


load('loss.mat')




figure
plot(loss)
for i=1:1:100
    ID(i,1) = i*10;
    LOSS(i,1) = loss(i*10);
end

