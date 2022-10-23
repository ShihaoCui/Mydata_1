% sz = size(H);
% for i=1:1:sz(2)
%     for j=1:1:i
%         HH(i) = sum(H(1:i));
%     end
% end
% hh1 = [5.1,11.4,19.9,24.7,30];
% vs1 = [150,205,350,250,350];
% 
% hh1 = HH;
% vs1 = VS;
% hh1 = [0,hh1];
% vs1 = [vs1,vs1(end)];
% plot(hh1,vs1)
% stairs(hh1,vs1)
% scatter(hh1,vs1)

clear;
clc;

H_all(1) = 2;
H_all_sum(1) = H_all(1);
L = 8;
for i =2:1:L
    H_all(i) = H_all(i-1)*1.2;
    H_all_sum(i) = sum(H_all(1:i));
end
H_all_sum(end)