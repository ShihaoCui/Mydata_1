clear all; clc; close all;

% 示例数据
load('SimSignals.mat')
t = [1:1:3000]/1e5;
S1 = S1./max(S1);
S2 = S2./max(S1);
% x = linspace(0, 2*pi, 100);
% y = sin(x);

filename = "UltrasoundGif.gif";
% 创建一个新的图形窗口
figure;

% 循环展示每个时间点的数据
for i = 1:50:length(t)
    plot(t(:,1:i), S1(:,1:i), 'r', 'LineWidth', 2);
    hold on
    plot(t(:,1:i), S2(:,1:i), 'b', 'LineWidth', 2);
    legend("S_1","S_2")
    title(sprintf('Time(s): %.4f', t(i)));
    xlabel('Time(s)');
    ylabel('Normalized Amplitude');
    axis([0, max(t), -1, 1]); % 设置坐标轴范围
    grid on;
    drawnow; % 更新图形
    pause(0.05); % 暂停一段时间，控制展示速度
    
    % 保存每一帧图像
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % 将每一帧图像保存到 GIF 文件中
    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
