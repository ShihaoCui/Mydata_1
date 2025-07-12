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
X = [cs2_all_ALL,Cutoff_Freq_all,DISP_Selected'./cs2_all_ALL];

X = X';


% Y = cs1_all_ALL;
Y = (H_all_ALL-min(H_all_ALL))./(max(H_all_ALL)-min(H_all_ALL));
% Y = [H_all_ALL,cs1_all_ALL];
% Y = (Y-min(Y))./(max(Y)-min(Y));
Y = Y';


save DataLoad X Y






train_no = floor(0.9*size(X,2));
% 假设 X_train 是输入数据，Y_train 是目标标签
X_train = X(:,1:train_no);  % 10个特征，100个训练样
Y_train = Y(:,1:train_no);   % 对应的目标输出

% 创建一个前馈神经网络
% net = feedforwardnet(20);  % 创建一个有10个神经元的隐藏层

% 创建一个包含两个隐藏层的神经网络，分别包含 10 和 5 个神经元
net = feedforwardnet([50]);

% % % 设置第一个隐藏层的激活函数为 'relu'
% net.layers{1}.transferFcn = 'tansig';
% % % % 
% % % % % 设置第二个隐藏层的激活函数为 'tansig'
% net.layers{2}.transferFcn = 'poslin';
% % net.layers{3}.transferFcn = 'poslin';
% % 







% 训练神经网络
net = train(net, X_train, Y_train);

% 假设 X_test 是新的输入数据
X_test = X(:,train_no+1:end);  % 5个新的样本
Y_test = Y(:,train_no+1:end);

% 使用训练好的网络进行预测
Y_pred = sim(net, X_test);

% 输出预测结果
disp('预测结果:');
1-mean(abs(Y_pred-Y_test)./Y_test,2)


%% FEM valivation
NN = 3;
feature1 = X_test(:,NN);
Y_pred1 = sim(net, feature1);
Y_pd = Y_pred1;
y_real = Y_test(NN);
1-abs(Y_pd - y_real )./y_real

%% FEM valivation
load('feature_FEM.mat')
feature1 = feature_FEM';
Y_pred1 = sim(net, feature1);
Y_pd = Y_pred1*(max(H_all_ALL)-min(H_all_ALL))+min(H_all_ALL);
y_real = 1/1000;
1-abs(Y_pd - y_real )./y_real
