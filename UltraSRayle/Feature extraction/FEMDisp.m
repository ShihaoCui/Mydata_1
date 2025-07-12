clear all;
close all;
clc

load('dataDigital.mat')
load('disp_extracted.mat')


% fre1 = fplot(:,1);

fre1 = dataDigital(:,1);
Cph_exp = dataDigital(:,2);


fre = fre1(fre1<700 & fre1>150);
Cph_exp = Cph_exp(fre1<700 & fre1>150);

DISP_exp = [fre,Cph_exp];



% DISP_exp([4,8],:) = [];

figure
hold on
scatter(DISP_exp(:,1),DISP_exp(:,2))
plot(dataDigital(:,1),dataDigital(:,2),'linewidth',3.5)
legend("Experimental","Anylitical")


xlim([0 1000])
ylim([2.5 6])



save DISP_exp DISP_exp


figure
load('DISP_exp.mat')
hold on
scatter(DISP_exp(:,1),DISP_exp(:,2))
plot(dataDigital(:,1),dataDigital(:,2),'linewidth',3.5)
legend("Experimental","Anylitical")
xlim([0 1000])
ylim([2.5 6])


freq =DISP_exp(:,1);
data = DISP_exp(:,2);


% 定义目标频率间隔，假设你想从第一个频率到最后一个频率以10Hz为间隔
target_freq = min(freq):10:max(freq);

% 对数据进行插值
interp_data = interp1(freq, data, target_freq, 'linear');  % 线性插值

% 绘制原始数据与插值结果
figure;
plot(dataDigital(:,1),dataDigital(:,2),'linewidth',4.5)
hold on;
scatter(target_freq, interp_data);  % 插值后的数据
xlabel('Frequency (Hz)');
ylabel('Data');
legend show;
legend("Experimental","Anylitical")

xlim([0 850])
ylim([2.5 6])

cs2  = 3.13;
feature_FEM  = [cs2, 204, interp_data(:,1:35)*1./cs2];

save feature_FEM feature_FEM

