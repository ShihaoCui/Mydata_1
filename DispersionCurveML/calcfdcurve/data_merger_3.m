load('Data11.mat')
dpr1 = Dpr;
vs1 = Layer_Config_Vs_Samples;
load('Data12.mat')
dpr2 = Dpr;
vs2 = Layer_Config_Vs_Samples;

clear Dpr;
clear Layer_Config_Vs_Samples;

DPR = [dpr1;dpr2];
Vs = [vs1;vs2];
clear dpr1 dpr2 vs1 vs2

save DPRVs DPR Vs
% save Vs1w Vs