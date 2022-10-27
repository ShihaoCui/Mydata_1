clear
clc

load('Disp_curve_exp.mat')
scatter(Disp_curve_exp(:,1),Disp_curve_exp(:,2))

f=Disp_curve_exp(:,1)';
VS=[200 300 400 300 350 600];
H=[5.1 6.3 8.5 4.8 5.3];
den=[1 1 1 1 1 1]*1800;
VP=VS*2;

% multi mode of Rayleigh wave
pvrl=calcbase(f,VS,H,VP,den);
hold on 
plot(f,pvrl')
legend("exp","Sim")