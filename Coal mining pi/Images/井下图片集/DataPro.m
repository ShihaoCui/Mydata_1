data = x0__1_;
figure()
image(data)% 原始图片

tt = 1;
DD = data(tt:10:1010,tt:10:1010,:);

figure(2)
imshow(DD)% 处理后的图片
% save DD DD