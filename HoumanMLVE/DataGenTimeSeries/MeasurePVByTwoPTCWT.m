function [E,freq,v,CClogram,wtA, wtB] = MeasurePVByTwoPTCWT(A,B,r,dt,FreqRange, VelocityRange, wavename)
%   Summary of this function goes here.
%   [E,freq,v,CClogram,wtA, wtB] = MeasurePVByTwoPTCWT(A,B,r,dt,FreqRange, VelocityRange, wavename)
%   Detailed explanation goes here.
%   The function is for measuring the phase velocity of dispersive waves from two traces.
%
%   IN      
%             A: the record of the first seismic trace, it's a column or raw vector. 
%             B: the sampling interval in time domain (s).
%             r: the distance (m) of A and B.
%            dt: the sampling interval in time domain (s).
%     FreqRange: the range of frequency (Hz) in the dispersive energy, such
%                as [5 100 1].
% VelocityRange: the parameter of the phase velocity (m/s), first is the 
%                minimal, second is the maximal, third is the interval, such 
%                as [50 800 1].
%      wavename: the type of wave, 'cmor1-1', 'db1','harr', and so on.
%
%  OUT   
%             E: the matrix of the normalized dispersive energy.
%          freq: the frequency (Hz) vecotor of the normalized dispersive energy.
%             v: the phase velocity (m/s) vector of the normalized dispersive energy.
%      CClogram: the cross-correlogram between A's cwt result and B's cwt result.
%           wtA: the cwt result of record A.
%           wtB: the cwt result of record B.
%  
%  References: 
%    Kijanka P , Ambrozinski L , Urban M W . Two Point Method For Robust Shear Wave 
%    Phase Velocity Dispersion Estimation of Viscoelastic Materials[J]. Ultrasound 
%    in Medicine & Biology, 2019, 45(9):2540-2553.
%
%  Author(s): Yan Yingwei
%  Copyright: 2020-2025 
%  Revision: 1.0  Date: 9/28/2020
%
%  Department of Earth and Space Sciences, Southern University of Science 
%  and Technology (SUST).


% 参数判断
if nargin==6
    wavename = 'cmor1-1';
elseif nargin==5
    wavename = 'cmor1-1';
    VelocityRange = [50 2000 5];
elseif nargin==4
    wavename = 'cmor1-1';
    VelocityRange = [50 2000 5];
    FreqRange = [5 100 1];
end

% 确定基本参数
vmin = VelocityRange(1);
vmax = VelocityRange(2);
dv = VelocityRange(3);
v = vmax:-dv:vmin;
lv = length(v);

fmin = FreqRange(1);
fmax = FreqRange(2);
df = FreqRange(3);
freq = fmin:df:fmax;
lf = length(freq);

Fs = 1/dt;      % 信号的采样率
M  = length(A);
E = zeros(lv,lf);
CClogram = zeros(lf,M);
t = dt:dt:M*dt;  % 信号的时间轴向量
ti = r./v;       % 由速度范围确定的互相关函数的时间范围

% 对信号作小波变换
wave_centfreq =centfrq(wavename);
scales = Fs*wave_centfreq./freq;

wtA = cwt(A,scales,wavename);
wtB = cwt(B,scales,wavename);

% 互相关得到频散能量矩阵和互相关谱
for i=1:lf
    temp = xcorr(wtB(i,:),wtA(i,:));
    CClogram(i,:) = temp(M:end);
    E(:,i) = interp1(t, real(CClogram(i,:)),ti,'PCHIP');
    E(:,i) = E(:,i).^(2).* E(:,i);
    E(:,i) = E(:,i)./max(abs(E(:,i)));
end
end

