% 示例数据
x = linspace(0, 2*pi, 100);
y = sin(x);

% 创建一个新的图形窗口
figure;

% 循环展示每个时间点的数据
for i = 1:length(x)
    plot(x(1:i), y(1:i), 'r', 'LineWidth', 2);
    title(sprintf('Time: %.2f', x(i)));
    xlabel('Time');
    ylabel('Value');
    axis([0, 2*pi, -1, 1]); % 设置坐标轴范围
    grid on;
    drawnow; % 更新图形
    pause(0.1); % 暂停一段时间，控制展示速度
    
    % 保存每一帧图像
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % 将每一帧图像保存到 GIF 文件中
    if i == 1
        imwrite(imind, cm, 'animation.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, 'animation.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
