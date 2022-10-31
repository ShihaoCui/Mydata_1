load('Data_F.mat')

Njc = 1;
for i=1:1:100000
    if JSF(i)<0.06
        DPRF_0_06(Njc,:) = DPRF(i,:);
        VSF_0_06(Njc,:) = VSF(i,:);
        Njc = Njc+1;
    end
end
Njc
min(JSF)

save Data_F_0_06 DPRF_0_06 VSF_0_06
