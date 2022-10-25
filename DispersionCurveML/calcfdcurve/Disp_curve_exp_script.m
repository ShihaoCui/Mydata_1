clear
clc

load('Disp_curve_exp.mat')
scatter(Disp_curve_exp(:,1),Disp_curve_exp(:,2))

f=Disp_curve_exp(:,1)';
VS=[199 300 350 380 400 600];
H=[5.1 6.3 8.5 4.8 5.3];
den=[1 1 1 1 1 1]*1000;
VP=VS*((3)^(1/1));

% multi mode of Rayleigh wave
pvrl=calcbase(f,VS,H,VP,den);
hold on 
plot(f,pvrl')
legend("exp","Sim")