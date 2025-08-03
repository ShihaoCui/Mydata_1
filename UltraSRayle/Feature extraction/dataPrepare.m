clear all;
close all;
clc;
rng(0);

load('DATA_ALL_ML.mat')

selected_fre_range  = round([1:20:700]);

for i=1:1:length(DISP_ALL_cutoff)
   DISP_Selected(:,i) = DISP_ALL_cutoff{i, 1}(round(selected_fre_range),end); %800/2 Hz range
   Fre_Selected(:,i) = DISP_ALL_cutoff{i, 1}(round(selected_fre_range),1); %800/2 Hz range 
end

X = [cs2_all_ALL,Cutoff_Freq_all,DISP_Selected'];

figure
nn  = 10;
plot(Fre_Selected(:,nn),DISP_Selected(:,nn))
hold on
plot(ones(size(Fre_Selected,2))*cs2_all_ALL(nn)*1.003)


f_min_ALL = Fre_Selected(1,:);


% Y = cs1_all_ALL;
% Y = (H_all_ALL-min(H_all_ALL))./(max(H_all_ALL)-min(H_all_ALL));
Y = [cs1_all_ALL,H_all_ALL];
% Y = (Y-min(Y))./(max(Y)-min(Y));

Input = [cs1_all_ALL,cs2_all_ALL,H_all_ALL] ;
Output = [f_min_ALL',cs2_all_ALL,DISP_Selected'];

% save ForwardData Input Output

% hold on
% plot(Cutoff_Freq_all)
% save DataLoad X Y

figure
scatter(Fre_Selected(:,2295),DISP_Selected(:,2295))
hold on
plot(Fre_Selected(:,2298),DISP_Selected(:,2298))
legend("Exp","Any")

