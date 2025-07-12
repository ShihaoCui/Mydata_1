clear all
close all
clc

% load('SAny.mat')
% data = SAny;
% load('data1_filted.mat')

load('data5.mat')
data1  = data;


% 假设 data1 是一个 2002x11 的矩阵
n_original = size(data1, 1);  % 原始行数
scale_factor  = 1;
n_new = round(n_original * scale_factor);  % 新的行数

% 创建一个新的向量来存储插值后的数据
data1_interp = zeros(n_new, size(data1, 2));

% 对每一列进行插值
for col = 1:size(data1, 2)
    % 创建原始的 x 值和新的 x 值
    x_original = 1:n_original;  % 原始的 x 值
    x_new = linspace(1, n_original, n_new);  % 新的 x 值，等间隔

    % 使用线性插值方法插值
    data1_interp(:, col) = interp1(x_original, data1(:, col), x_new, 'linear');
end


data1 = data1_interp;

% data = zeros(size(data1));
% sss = 500;
% data(1:sss,1:11) = data1(1:sss,1:11);
% data = data1(1:2000,5:2:11);% 结果不错
data = data1(1:2000*scale_factor,6:1:13);% 结果不错
% data = data1(1:2000,1:2:10);% 结果不错


%%
% ------------- Dispersion analysis ---------------
%%
% Filename = 'SampleData.dat';
HeaderLines = 7;
fs = 1e5*scale_factor; % Hz % 采样频率
N = size(data,2); %% 数据的列数
x1 = 8/1000; % m %%  第一个 接收器的 位置
dx = 0.5/1000; % m %% 两个 接收器的 距离
% Length of receiver spread [m]
L = (N-1)*dx; 

% [u,T,Tmax,L,x] = MASWaves_read_data(Filename,HeaderLines,fs,N,dx,x1,Direction);
u = data(1:end,1:end)./max(max(data))*1e-5;
% Time of individual recordings [s]
% T = data1(:,1)./1e6; 
T = (1:size(u,1))./fs;
% Total recording time [s]
Tmax = max(T);
% Location of receivers, distance from seismic source [m]
x = (x1):dx:(L+x1);

%%
du = 1/75;
FigWidth = 6; % cm
FigHeight = 8; % cm
FigFontSize = 12; % pt

figure
MASWaves_plot_data(u,N,dx,x1,L,T,Tmax,du,FigWidth,FigHeight,FigFontSize)

%%
cT_min = 0.01; % m/s
cT_max = 8; % m/s
delta_cT = 0.02; % m/s

[f,c,A] = MASWaves_dispersion_imaging(u,N,x,fs,cT_min,cT_max,delta_cT);



%%
resolution = 1;
fmin = 10; % Hz
fmax = 1500; % Hz
FigWidth = 7; % cm
FigHeight = 7; % cm
FigFontSize = 8; % pt
figure
[fplot,cplot,Aplot] = MASWaves_plot_dispersion_image_2D(f,c,A,fmin,fmax,...
    resolution,FigWidth,FigHeight,FigFontSize);



figure
imagesc(fplot(:,1),cplot(1,:),Aplot')
axis xy
set(gca, 'YDir', 'normal')

for i=2:1:size(Aplot,1)
    NN = 20;
    temp = Aplot(i,NN:end);
    z = find(temp==max(temp));
%     Cph_exp(i) = cplot(1,min(z+NN,size(Aplot,2)));
    Cph_exp(i) = cplot(1,z+18);
end

load('dataDigital.mat')
figure
hold on
scatter(fplot(:,1),Cph_exp)
plot(dataDigital(:,1),dataDigital(:,2),'linewidth',3.5)
legend("Experimental","Anylitical")
xlim([0 1000])



figure
imagesc(fplot(:,1),cplot(1,:),Aplot')
% colormap(jet)
hold on
scatter(fplot(:,1),Cph_exp)
plot(dataDigital(:,1),dataDigital(:,2),'r-','linewidth',2.5)
axis xy
set(gca, 'YDir', 'normal')
xlim([0 1000])
% save disp_extracted fplot Cph_exp