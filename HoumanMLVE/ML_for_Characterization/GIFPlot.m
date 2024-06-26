clear all;
close all;
clc;

filename = 'DataTrainViscoElatic1.gif'; % GIF文件名
load('DataTrainViscoElatic1.mat')
images = PicData;
noise = 0.3*rand(size(images)); 
% 将噪声加到原始数据上
images = images + noise;

vectors = LabelAll;
vectors(:,1) = vectors(:,1)/1e3;
for i = 1:100
    % 创建一个图形窗口
    figure('visible', 'on'); % 隐藏图形窗口以加快处理速度
    % 子图1：展示图片
    subplot(1, 2, 1);
%     imshow(squeeze(images(:, :, i))); % 假设图片是灰度的
    imagesc([100 500],[0 10],images(:, :, i)');
    colormap(jet);
    colorbar;
    set(gca,'YDir','normal');
    xlabel('Frequency(Hz)');
    ylabel('Phase velosity (m/s)');
    titleText = ['DPR Image: ', num2str(i)];
    title(titleText);

    
    % 子图2：展示向量的柱状图
    subplot(1, 2, 2);
    bar(vectors(i, :));
    set(gca, 'xticklabel', {"\mu_1 (KPa)", "\mu_2 (Pa·s)"});
    title('Viscoelastic Properties');
    ylim([0, max(vectors(:))]); % 假设你想要所有柱状图有相同的Y轴范围
    
    % 捕获并保存当前帧
    frame = getframe(gcf);
    img = frame2im(frame);
    [imind,cm] = rgb2ind(img, 256);
    
    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append');
    end
    
    close; % 关闭当前图形窗口
end
