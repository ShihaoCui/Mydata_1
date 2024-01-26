clear all;
clc;
close all;

load('Data2.mat')
Data3 = Data2;

n=0;
m=0;
for i=1:1:size(Data2,1)
    if (Data2(i,2)>0.2) || (Data2(i,2)<0)
        Data3(i,2) = 0;
        n=n+1;
    elseif (Data2(i,2)>=0) && (Data2(i,2)<=0.2)
        Data3(i,2) = 1;
        m=m+1;
    end
end


