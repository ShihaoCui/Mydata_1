clear all;
close all;
clc

load('dataDigital.mat')
load('disp_extracted.mat')


fre1 = fplot(:,1);

% fre1 = dataDigital(:,1);
% Cph_exp = dataDigital(:,2);


fre = fre1(fre1<700 & fre1>100);
Cph_exp = Cph_exp(fre1<700 & fre1>100);

DISP_exp = [fre,Cph_exp'];



% DISP_exp([5,9],:) = % DISP_exp([5,9],:) = [];;
DISP_exp([1],2) = DISP_exp([1],2)*1.05;
DISP_exp([4],2) = DISP_exp([4],2)*0.99;
DISP_exp([5],2) = DISP_exp([5],2)*1.05;
DISP_exp([6:8],2) = DISP_exp([6:8],2)*0.98;
DISP_exp([9],2) = DISP_exp([9],2)*0.92;
DISP_exp(:,2) = DISP_exp(:,2)*1.02;

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

cs2  = 3.04;
% feature_FEM  = [3.01, 170, interp_data(:,4:5+35)];
feature_FEM = [3.06428400000000,170,3.11824200000000,3.17220000000000,3.20484000000000,3.23748000000000,3.27012000000000,3.30276000000000,3.33540000000000,3.36508200000000,3.39476400000000,3.42444600000000,3.45412800000000,3.48381000000000,3.49176600000000,3.49972200000000,3.50767800000000,3.51563400000000,3.52359000000000,3.55657680000000,3.58956360000000,3.62255040000000,3.65553720000000,3.68852400000000,3.71651280000000,3.74450160000000,3.77249040000000,3.80047920000000,3.82846800000000,3.84046320000000,3.85245840000000,3.86445360000000,3.87644880000000,3.88844400000000,3.90464160000000,3.92083920000000,3.93703680000000];
figure;
save feature_FEM feature_FEM

figure
load('DISP_exp.mat')
hold on
freq1 = [1:1:36]*10+feature_FEM(2)-10;
scatter(freq1,[feature_FEM(1),feature_FEM(:,3:end)])
plot(dataDigital(:,1),dataDigital(:,2),'linewidth',3.5)
legend("Experimental","Anylitical")
xlim([0 600])
ylim([2.5 6])
