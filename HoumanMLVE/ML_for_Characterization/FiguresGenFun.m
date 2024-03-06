function  pic= FiguresGenFun(u1, u2,rho, freq,v)

pic = zeros(length(freq),length(v));
Vs= DPRFun(u1, u2,rho, freq);
for i=1:1:length(freq)
      for j=1:1:length(v)
           [~,Index]= min(abs(v-Vs(i)));
           pic(i,Index) = 1;
      end
end
    
end