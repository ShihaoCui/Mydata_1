clear all;
close all;
clc;

% 参数设置
minx = 0;
maxx = 1;
miny = 0;
maxy = 1;
mint = 0;
maxt = 0.03;
dt = 1e-5;
dx = 0.01;
dy = 0.01;
f0 = 400; % 激励频率

% 网格尺寸
dimt = round((maxt - mint) / dt) + 1;
dimx = round((maxx - minx) / dx) + 1;
dimy = round((maxy - miny) / dy) + 1;

% 生成网格
x = linspace(minx, maxx, dimx);
y = linspace(miny, maxy, dimy);
t = linspace(mint, maxt, dimt);

% 波速分布（示例，需要根据实际情况修改）
v = ones(dimx, dimy) * 35; % 这里假设波速为2，实际中应根据位置变化
% v(40:60,20:40) = 50;

figure
imagesc(v')
colorbar;

% 初始化波动函数 u
u = zeros(dimx, dimy, dimt);

% 激励函数
b = 50;
att = exp(-b*t);
xx = att.*sin(2 * pi * f0 * t);
figure
plot(t,xx)

eta0 = ones(dimx, dimy)*2.2e-3;
% eta0(40:60,20:40) = 3.2e-3;

rho = ones(dimx, dimy);
% 更新规则 - 有限差分法
for k = 2:dimt-1
    for i = 2:dimx-1
        for j = 2:dimy-1
            if i == 2 && j == 2
                u(i, j, k+1) = xx(k); % 激励点
            else
                starX = (u(i+1, j, k)+u(i+1, j, k-1))+(u(i-1, j, k)+u(i-1, j, k-1))-2*(u(i, j, k)+u(i, j, k-1));
                starX = starX./(dx^2*dt);
                starY = (u(i, j+1, k)+u(i, j+1, k-1))+(u(i, j-1, k)+u(i, j-1, k-1))-2*(u(i, j, k)+u(i, j, k-1));
                starY = starY./(dy^2*dt);
                star = eta0(i,j)/rho(i,j)*dt^2*(starX+starY);
                u(i, j, k+1) = 2 * u(i, j, k) - u(i, j, k-1) + ...
                              (v(i, j)^2 * dt^2 / dx^2) * (u(i+1, j, k) - 2*u(i, j, k) + u(i-1, j, k)) + ...
                              (v(i, j)^2 * dt^2 / dy^2) * (u(i, j+1, k) - 2*u(i, j, k) + u(i, j-1, k));
                u(i, j, k+1) = u(i, j, k+1)+star;
            end
        end
    end
    % 边界条件：位移为0
    u(1, :, k+1) = 0;
    u(end, :, k+1) = 0;
%     u(:, 1, k+1) = 0;
    u(:, end, k+1) = 0;
end

% 可视化
figure

for k = 1:100:dimt
    set(gca,'ZLim',[-5,5]);
    imagesc(x,y,v'/max(v(:)));
    colormap('jet');
    hold on
    imagesc(x, y, u(:,:,k)'/max(u(:)),'AlphaData', 0.5)
    colormap('jet');
%     colorbar;
    zlim([-1, 1]);
    title(sprintf('Time = %.4f s', t(k)));
    drawnow;
    
%     sprintf('Time = %.2f s', t(k))
    fr = getframe;
%     text(0.8,0.9,0.95,'2DWave','fontsize',10,'color','b')
%     text(0.8,0.9,0.9,sprintf('Time = %.4f s', t(k)),'fontsize',10,'color','b')
    im=frame2im(fr);

    [I,map] = rgb2ind(im,256);

    if k==1

        imwrite(I,map,'Wave.gif','gif','loopcount',inf,'Delaytime',0.0001)

    else

        imwrite(I,map,'Wave.gif','gif','writemode','append','Delaytime',0.0001)

    end
    
end

% figure
% S1 = u(11,2,:);
% S1 = reshape(S1,length(t),1);
% plot(t,S1)
% hold on 
% S1 = u(21,2,:);
% S1 = reshape(S1,length(t),1);
% plot(t,S1)
% legend("S1","S2")
