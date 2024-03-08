clear all;
close all;
clc;

u1_max = 25*10^3; %Kpa 25
u1_min = 1*10^3; % Kpa 1

u2_max = 10; % Pa*s 10
u2_min = 1; % Pa*s 1

thrRang = [0.98 0.99];
freq = [10:1:50]*10;
v = [1:1:50]*0.2;

% Time start
tic;

SampleNo = 1;
SampleNoMax = 5000;

PicData = [];
LabelAll = [];% [u1,u2]
VsAnyAll = [];

while SampleNo <=SampleNoMax
u1 = rand*(u1_max-u1_min)+u1_min;
u2 = rand*(u2_max-u2_min)+u2_min;
rho = 1000;


thr=(thrRang(2)-thrRang(1))*rand+thrRang(1);

pic= FiguresGenFun(u1, u2,rho, freq,v,thr);
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
disp(['Time needed：', num2str(elapsed_time/60), ' mins']);

% save DataTrainViscoElatic PicData LabelAll VsAnyAll
% '-v7.3'