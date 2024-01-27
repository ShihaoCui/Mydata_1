clear all;
clc;
close all;

load('Data4.mat')
% scatter([1:1:48],Data4(:,1));

sd = Data4(:,1);

% size(find(sd<0.01 & sd>=0.0))

sd1 = 1-sd/0.2;
Data4_1 = Data4;

Data4_1(:,1) = sd1;


% size(find(sd1<0.50))

Data4_2 = Data4_1;

for i=1:1:48
    if (Data4_1(i,1)>=0.95 )
        Data4_2(i,1) = 0;
    elseif (Data4_1(i,1)>=0.90 && Data4_1(i,1)<0.95)
        Data4_2(i,1) = 1;
    elseif (Data4_1(i,1)>=0.80 && Data4_1(i,1)<0.90)
        Data4_2(i,1) = 2;
    elseif (Data4_1(i,1)>=0.70 && Data4_1(i,1)<0.80)
        Data4_2(i,1) = 3;
    elseif (Data4_1(i,1)>=0.60 && Data4_1(i,1)<0.70)
        Data4_2(i,1) = 4;
    elseif (Data4_1(i,1)<0.60)
        Data4_2(i,1) = 5;
    end
end

save Data4_2 Data4_2 