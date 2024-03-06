clear all;
close all;
clc;

u1_max = 25*10^3; %Kpa
u1_min = 1*10^3; % Kpa

u2_max = 1; % Pa*s
u2_min = 10; % Pa*s

freq = [10:1:50]*10;
v = [1:1:50]*0.2;

% Time start
tic;

SampleNo = 1;
SampleNoMax = 10000;
PicData = [];
LabelAll = [];% [u1,u2]
VsAnyAll = [];

while SampleNo <=SampleNoMax
u1 = rand*(u1_max-u1_min)+u1_min;
u2 = rand*(u2_max-u2_min)+u2_min;
rho = 1000;

pic= FiguresGenFun(u1, u2,rho, freq,v);
PicData(:,:,SampleNo) = pic;
LabelAll(SampleNo,:) = [u1,u2];

Vs= DPRFun(u1, u2,rho, freq);
VsAnyAll(SampleNo,:) = Vs;

SampleNo = SampleNo+1;

% figure
% imagesc(freq,v,pic');
% colormap(jet);
% colorbar;
% set(gca,'YDir','normal');
% title('CWT DPR');
% xlabel('Fre(Hz)');
% ylabel('Phase velosity (m/s)');
% hold on
% plot(freq,Vs,'LineWidth', 5);

end


% Time elapse and output
elapsed_time = toc;
disp(['Time needed：', num2str(elapsed_time), ' s']);

save DataTrainViscoElatic PicData LabelAll VsAnyAll '-v7.3'