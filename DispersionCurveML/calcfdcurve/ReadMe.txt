 References:
凡友华.层状介质中瑞利面波频散曲线正反演研究[D].哈尔滨工业大学，2001.
Please read the statement.pdf.

Main program for Surface Wave dispersive curve.
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



%% Demo
f=0:50:5000;
%% 两层递增型地质模型
VS=[500 2000];
H=[0.2];
den=[1700 2000];
VP=[1200 4000];
% base mode of Rayleigh wave
pvrl=calcbase(f,VS,H,VP,den);
% base mode of Love Wave.
pvlv=calclovebase(f,VS,H,den);

% Please run the mainpro.m
>>mainpro

% if you have any problems, do not hesitate to contact with me.
% Email: wallace2012y@outlook.com
% Tel: 18844193600
% WeChat: Groundfrog


