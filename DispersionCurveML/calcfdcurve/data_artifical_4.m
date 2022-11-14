clear;
clc;

% close all;
load('dpc_26_Intp_all.mat');
dpc = a(:,2,:);
DPC26 = reshape(dpc,26,33);
DPC26sz = size(DPC26);
DPC26sz1 = DPC26sz(1);

L = 10;
Err_min = 0.09;
% density
den = ones(1,L+1);
den = 1800*den;


% frequency
f = [7:1:39];

% H
for i=1:1:L
    H(i) = 3;
end

NNN = 1;
N_Samples = 50000

VsF = [];
Dpr = [];
JS_Samples = [];
while NNN<=N_Samples
% VS
V1 = 200*(1+0.3*rand);
V2 = 250*(1+0.3*rand);
V3 = 400*(1+0.3*rand);
V4 = 350*(1-0.1*rand);
V5 = 400*(1+0.2*rand);

% V1 = 200;
% V2 = 300;
% V3 = 350;
% V4 = 400;
% V5 = 300;

% V1 = 200;
% V2 = 250;
% V3 = 400;
% V4 = 350;
% V5 = 450;

VS = [   V1 V1*(1+0.05*rand) ...
         V2 V2*(1+0.05*rand) ...
         V3 V3*(1+0.05*rand) ...
         V4 V4*(1-0.05*rand) ...
         V5 V5*(1+0.1*rand) ...
         600];
     
for i=1:1:L
    H_all(i) = sum(H(1:i));
end

% figure()
stairs([0,H_all]',VS(1,1:end))


% VP
VP = VS*2;

% base mode of R waves
pvrl=calcbase(f,VS,H,VP,den);


 % judgement critera for pre-selection
        jc = mean(mean(abs(DPC26-pvrl)./DPC26));
        
        if jc<Err_min
        %         Layer_Config_Samples = [Layer_Config_Samples;Layer_config]
            VsF = [VsF;VS];
            Dpr =[Dpr;pvrl];
            JS_Samples = [JS_Samples;jc];
            NNN = NNN+1
%         else
%             contine
        end


% figure
% plot(f,pvrl)
% hold on
% % load('Disp_curve_exp.mat')
% load('dpc_26_Intp_all.mat')
% 
% b = a(12,:,:);
% b = reshape(b,2,33);
% scatter(b(1,:),b(2,:))
% legend('sim','exp')
end

save Data_a4 VsF Dpr