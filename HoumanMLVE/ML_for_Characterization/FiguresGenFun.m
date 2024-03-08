function  pic= FiguresGenFun(u1, u2,rho,freq,v,thr)

% pic = ones(length(freq),length(v));
% Vs= DPRFun(u1, u2,rho, freq);
% for i=1:1:length(freq)
%       for j=1:1:length(v)
%            [~,Index]= min(abs(v-Vs(i)));
%            pic(i,Index) = 1;
%       end
% end


pic = zeros(length(freq),length(v));
Vs= DPRFun(u1, u2,rho, freq);
for i=1:1:length(freq)
      for j=1:1:length(v)
           DD = (abs(v-Vs(i)));
           DD2 = 1-(DD-min(DD))./(max(DD)-min(DD));
           DD2(DD2<thr) = 0;
           pic(i,:) = DD2;
      end
end
    
end