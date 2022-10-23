clear;
clc;

tic
N_sample = 100;
F_i = [ 6, 8, 10,13,15.3,18.7,21.4,25,27.9,31.4];
sz = size(F_i);
error_zero = zeros(1,sz(2));

Dpr = [];
Layer_Config_Samples = [];

for n=1:1:N_sample
     [H, VS, VP, den] = Layers_set_gen_Markov_fun();
     Layer_config = [H, VS, VP, den];
%      base mode of Rayleigh wave
     try
        i=1:1:sz(2);
        f = F_i(i);    
        pvrl=calcbase(f,VS,H,VP,den);
        Layer_Config_Samples = [Layer_Config_Samples;Layer_config];
        Dpr =[Dpr;pvrl];
     catch
         continue
% 
     end
    
end

toc

% save Data11 Dpr Layer_Config_Samples
