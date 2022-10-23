function  plotbase(f,pv)
plot(f,pv,'k-','linewidth',1);
xlabel('frequency Hz');
ylabel('phase velocity/m*s^{-1}');
grid on;
%hold on;
%axis([min(f),max(f), min(pv)*0.7, max(pv)]);
% axes('GridLineStyle', '--')
end

