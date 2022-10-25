clear;
clc;

tic
N_sample = 50000;
F_i = [7:1:39];
sz = size(F_i);
error_zero = zeros(1,sz(2));

Dpr = [];
Layer_Config_Samples = [];
Layer_Config_Vs_Samples = [];

for n=1:1:N_sample
     [H, VS, VP, den] = Layers_set_gen_Markov_fun_1();
     Layer_config = [H, VS, VP, den];
%      base mode of Rayleigh wave
     try
        i=1:1:sz(2);
        f = F_i(i);    
        pvrl=calcbase(f,VS,H,VP,den);
%         Layer_Config_Samples = [Layer_Config_Samples;Layer_config];
        Layer_Config_Vs_Samples = [Layer_Config_Vs_Samples;VS];
        Dpr =[Dpr;pvrl];
     catch
         continue
% 
     end
    
end

toc
save Data12 Dpr Layer_Config_Vs_Samples
