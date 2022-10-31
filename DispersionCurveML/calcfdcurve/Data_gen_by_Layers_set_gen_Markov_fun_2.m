clear;
clc;
load('dpc_26_Intp_all.mat');
dpc = a(:,2,:);
DPC26 = reshape(dpc,26,33);
DPC26sz = size(DPC26);
DPC26sz1 = DPC26sz(1);


tic
N_sample = 100000;
F_i = [7:1:39];
sz = size(F_i);
error_zero = zeros(1,sz(2));

Err_min = 0.08;% accepted error rate

Dpr = [];
Layer_Config_Samples = [];
Layer_Config_Vs_Samples = [];
JS_Samples = [];

NNN=1;
while NNN<N_sample+1
% for n=1:1:N_sample
     [H, VS, VP, den] = Layers_set_gen_Markov_fun_1();
     Layer_config = [H, VS, VP, den];
%      base mode of Rayleigh wave

     try
        i=1:1:sz(2);
        f = F_i(i);    
        pvrl=calcbase(f,VS,H,VP,den);
        
        % judgement critera for pre-selection
        jc = mean(mean(abs(DPC26-pvrl)./DPC26));
        
        if jc<Err_min
        %         Layer_Config_Samples = [Layer_Config_Samples;Layer_config]
            Layer_Config_Vs_Samples = [Layer_Config_Vs_Samples;VS];
            Dpr =[Dpr;pvrl];
            JS_Samples = [JS_Samples;jc];
            NNN = NNN+1
%         else
%             contine
        end
       
     catch
         continue
    % 
     end
    
% end
end

toc

VSF = Layer_Config_Vs_Samples;
DPRF = Dpr;
JSF = JS_Samples;

save Data_F  VSF DPRF JSF



% for i =1:1:26
% Aa(i) = mean(mean(abs(DPC26-DPC26(i,:))./DPC26));
% end
