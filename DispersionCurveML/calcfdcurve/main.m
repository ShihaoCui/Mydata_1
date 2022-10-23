% Main program for Surface Wave dispersive curve.
%  Author(s): Yan Yingwei
%  Copyright: 2017-2019 
%  Revision: 1.0  Date: 3/7/2017

% This is the main program to calculate the analytical solution of Surface
% Wave dispersive curve.
% It involves function 
%  fastcalc.m: the Key function of calculating the analytical solution of 
%               Rayleigh Wave dispersive curve.
%  calcbase.m: calculate the base mode dispersive curve of Rayleigh Wave.
%  fastlovecalc.m: the Key function of calculating the analytical solution
%               of Love Wave dispersive curve.
% calclovebase.m: calculate the base mode dispersive curve of Love Wave.


f=0:50:5000;
%% 两层递增型地质模型
VS=[500 2000];
H=[0.2];
den=[1700 2000];
VP=[1200 4000];

% base mode of Rayleigh wave
%pvrl=calcbase(f,VS,H,VP,den);

% multi mode of Rayleigh wave
pvrlmulti=calcmulti(f,VS,H,VP,den);

% base mode of Love Wave.
%pvlv=calclovebase(f,VS,H,den);

% multi mode of Love wave

pvlvmulti=calclovemulti(f,VS,H,den);

figure(1)
plotbase(f,pvrlmulti(:,1));
axis([min(f),max(f), min(VS)*0.7, max(VS)]);
title('基模式瑞雷波频散曲线');

figure(2)
plotmulti(f,pvrlmulti);
axis([min(f),max(f), min(VS)*0.7, max(VS)]);
title('多模式瑞雷波频散曲线');

figure(3)
plotbase(f,pvlvmulti(:,1));
axis([min(f),max(f), min(VS)*0.7, max(VS)]);
title('基模式勒夫波频散曲线');

figure(4)
plotmulti(f,pvlvmulti);
axis([min(f),max(f), min(VS)*0.7, max(VS)]);
title('多模式勒夫波频散曲线');
