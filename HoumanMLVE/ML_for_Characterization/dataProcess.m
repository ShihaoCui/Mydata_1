clear all;
close all;
clc;

load('DataTrainViscoElatic.mat')
pics = PicData(:,:,1000);


%% Case 1 
% Real Physical properties
u1 = 2.3*10^3; % KPa
u2 = 2.2; % Pa*s
rho = 1000;

% range of Fre and Vs
freq = [10:1:50]*10;
v = [1:1:50]*0.2;

% DPR extracted from wavelet energy analysis
load('E_filter2.mat')
pic_sim2 = E_filter;
pic_sim2 = (fliplr(E_filter));

load('E_filter1.mat')
pic_sim1 = E_filter;
pic_sim1 = (fliplr(E_filter));



figure
subplot(1,2,1)
imagesc(freq,v,pic_sim1');
colormap(jet);
colorbar;
set(gca,'YDir','normal');
title('CWT DPR');
xlabel('Fre(Hz)');
ylabel('Phase velosity (m/s)');
hold on
Vs= DPRFun(u1, u2,rho, freq);
plot(freq,Vs,'LineWidth', 5);


pic= FiguresGenFun(u1, u2,rho, freq,v,0.95);
Vs= DPRFun(u1, u2,rho, freq);
subplot(1,2,2)
imagesc(freq,v,pic');
colormap(jet);
colorbar;
set(gca,'YDir','normal');
title('CWT DPR');
xlabel('Fre(Hz)');
ylabel('Phase velosity (m/s)');
hold on
plot(freq,Vs,'LineWidth', 5);

pics_sim(1,:,:) = [pic_sim1];
pics_sim(2,:,:) = [pic_sim2];
save pics_sim pics_sim