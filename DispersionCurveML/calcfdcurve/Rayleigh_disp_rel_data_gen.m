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
clear;
clc;

F_i = [6,8, 10,13,15.3,18.7,21.4,25,27.9,31.4];
sz = size(F_i);
N_samp = 500

Dpr = [];
Layers_Set = [];

tic
for n=1:1:N_samp
    
    i=1:1:sz(2);
    %% layer setup


    f = F_i(i);
%     H=[5 10 15];
%     VS=[200 300 500 1000];
%     VP=[400 600 1000 2000];
%     den=[1000 1200 1500 2000];
    
%     [H, VS, VP, den] = Layers_set_gen_fun();
    [H, VS, VP, den] = Layers_set_gen_Markov_fun();
    
    layers_set = [H,VS(1:end),VP(1:end),den(1:end)];

    % base mode of Rayleigh wave
    pvrl=calcbase(f,VS,H,VP,den);
    
    Lay_set = [];
    Dpr =[Dpr;pvrl];
    Layers_Set = [Layers_Set;layers_set];
    
end

toc

