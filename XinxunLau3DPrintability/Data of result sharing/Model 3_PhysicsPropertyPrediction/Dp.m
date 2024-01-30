load('Data5.mat')

Data5(:,1) = Data5(:,1)/1000;
Data5(:,4) = Data5(:,4)/1000;

save Data5 Data5
