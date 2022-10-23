function plotmulti(f,pvmulti)
[M,N]=size(pvmulti);



for k=1:N
    ind=0;
    for j=1:M
        if(pvmulti(j,k)~=0)
            ind=ind+1;
            f_temp(ind)=f(j);
        end
    end
    beg=M-ind+1;
    plot(f_temp,pvmulti(beg:1:M,k),'k-','linewidth',1);
    clear f_temp;
    hold on;
end
xlabel('频率/Hz');
ylabel('相速度/m·s^{-1}');
grid on;
hold on;
axis([min(f),max(f), min(pvmulti(:,1))*0.7, max(pvmulti(:,1))/0.7]);
end

