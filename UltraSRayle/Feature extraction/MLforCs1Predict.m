clear all;
close all;
clc;
rng(0);

load('DATA_ALL_ML.mat')

X = [cs2_all_ALL,Cutoff_Freq_all,H_all_ALL];
X = X';
Y = cs1_all_ALL;
% Y = [H_all_ALL,cs1_all_ALL];
% Y = (Y-min(Y))./(max(Y)-min(Y));
Y = Y';

train_no = floor(0.95*size(X,2));
% 假设 X_train 是输入数据，Y_train 是目标标签
X_train = X(:,1:train_no);  % 10个特征，100个训练样本
Y_train = Y(:,1:train_no);   % 对应的目标输出


















% 创建一个前馈神经网络
net = feedforwardnet([15]);  % 创建一个有10个神经元的隐藏层

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

%% FEM data validation
load('feature_FEM.mat')
feature1 = [feature_FEM(1),feature_FEM(2),1/1000]';
Y_pred1 = sim(net, feature1);
y_real = 9;
1-abs(Y_pred1-y_real)./y_real