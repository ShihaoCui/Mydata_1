clc
close all
clear all


%***********1.Input Harmonic waves***********%

fs =100000; %设定采样频率
N =3000;
n =0:N -1;
t = n/ fs;
f0 =400; %设定正弦信号频率

%生成正弦信号
b = 500;
att = exp(-b*t);
x = att.*sin(2*pi*f0*t);
figure(1);
plot(t,x); %作正弦信号的时域波形
xlabel('Time/ s');
ylabel('Amplitude');
title('Time Series Waveform');
grid;

%进行FFT 变换并做频谱图

y = fft(x,N); %进行FFT 变换
mag_im = imag(y); %求幅值
mag_re = real(y);
mag = abs(y);
f = (0:length(y) -1)'*fs/ length(y); %进行对应的频率转换
figure
plot(f,mag_im); %作频谱图
axis([0,2000,-1200,1200]);
xlabel('Frequency/ Hz');
ylabel('Mag');
title('Mag Spectrum');
grid;


%*********1. Transfer function*********%
u1 = 10*10^3; % KPa
u2 = 7; % Pa*s
rho=1000;
omega = 2*pi*f;

Vs = sqrt(2*(u1^2+omega.^2*u2^2)./(rho*(u1+sqrt(u1^2+omega.^2*u2^2))));
kr = omega./Vs;
ki = sqrt((rho*omega.^2.*(sqrt(u1^2+omega.^2*u2^2)-u1))./(2*(u1^2+omega.^2*u2^2)));
% ki = 1000;
% distance between two sensors: dx = 2/1000 .
r1 = 2/1000;
r2 = 2*r1;
dx = r2-r1;
% transfer function TF
TF = sqrt(r2/r1).*exp(i*kr*dx-ki*dx);
TF = TF';

%*********2. S2_w: reconstructed signals u1(w)*exp(ikx) *********%
S1_w = y;
S2_w = S1_w.*TF;
xifft = ifft(S2_w,'symmetric');
magx = real(xifft);
ti = [0:length(xifft)-1] / fs;
figure
plot(ti,magx);
xlabel('Time/ s');
ylabel('Mag');
title('IFFT-reconstructed time-series waveform');
grid;

S2 = magx;
S1 = x;
figure
plot(S1)
hold on
plot(S2)
legend("S1","S2")
save SimSignals87 S1 S2 u1 u2
% save S2_w S2_w