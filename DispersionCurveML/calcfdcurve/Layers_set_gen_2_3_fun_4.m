L = 10;

% H
for i=1:1:L
    H(i) = 3;
end

% density
den = ones(1,L+1);
den = 1800*den;

% frequency
f = [7:1:39];


%% case [2,2,2,2,2] five layers 
LL = 5;

VS1_min = 150;
VS1_max = 210;
VS_end_max = 600;
%    VS(end) = VS_end_max;
VLL(1) = rand*(VS1_max -VS1_min )+VS1_min;
 
lbd_min = 0.01;
lbd_max = 0.6;
   
for i =2:1:5
    lbd = (lbd_max-lbd_min)*rand+lbd_min;
    aa = (1)^(round(10*rand));
    VLL(i) = VLL(i-1)+(aa)*lbd*VLL(i-1);
    
    if VLL(i)<=200
        VLL(i)=VLL(i-1);
    end
    
    if VLL(i)>600
        VLL(i) = VLL(i-1);
    end
        
end


V1 = VLL(1);
V2 = VLL(2);
V3 = VLL(3);
V4 = VLL(4);
V5 = VLL(5);

% figure
% stairs([1,2,3,4,5],VLL)
% VS
% V1 = 200*(1+0.3*rand);
% V2 = 250*(1+0.3*rand);
% V3 = 400*(1+0.3*rand);
% V4 = 350*(1-0.1*rand);
% V5 = 400*(1+0.2*rand);


VS = [   V1 V1*(1+0.05*rand) ...
         V2 V2*(1+0.05*rand) ...
         V3 V3*(1+0.05*rand) ...
         V4 V4*(1-0.05*rand) ...
         V5 V5*(1+0.1*rand) ...
         600]    
for i=1:1:L
    H_all(i) = sum(H(1:i));
end

% figure()
% stairs([0,H_all]',VS(1,1:end))


% VP
VP = VS*2;

% base mode of R waves
pvrl=calcbase(f,VS,H,VP,den);

% figure
plot(f,pvrl)
hold on
% load('Disp_curve_exp.mat')
load('dpc_26_Intp_all.mat')

b = a(12,:,:);
b = reshape(b,2,33);
scatter(b(1,:),b(2,:))
legend('sim','exp')