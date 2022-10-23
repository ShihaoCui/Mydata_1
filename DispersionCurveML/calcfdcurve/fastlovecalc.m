function y= fastlovecalc(x)
%   Summary of this function goes here
%   function y=fastlovecalc(x)
%   The function is the key function for calculating the disperive function
%   of Love wave.
%   Detailed explanation goes here
%   The function is called by the function named calclovebase,calclovemulti.
%   The function would search the zero solution of fastlovecalc.
%    x: The possible phase velocity.
%
%   References: Schwab F A,Knopoff L.Fast surface wave and free mode
%               computions[M].In:Methods in Computational Physics.Bolt B
%               A(ed.).New York: Academic Press,1972:87-180.
%
%  Author(s): Yan Yingwei
%  Copyright: 2017-2019 
%  Revision: 1.0  Date: 3/7/2017.
%
%  Department of Geophysics, Jilin University.

%%  
global mode_base
f=mode_base(1);
[~,n]=size(mode_base);
n=n/3;
VS=mode_base(2:n+1);
H=mode_base(n+2:2*n);
den=mode_base(2*n+1:3*n);
k=2*pi*f/x;
s=-den(n)*VS(n)^2*sqrt((1-x^2/VS(n)^2));
T0=[s -1i];
F=T0;
a=zeros(2,2,n-1);
for i=1:n-1
    r=-1i*sqrt(1-x^2/VS(i)^2);
    h=H(i);
    u=den(i)*VS(i)^2;
    Q=k*r*h;
    a(:,:,i)=[cos(Q) 1i*sin(Q)/(u*r);1i*u*r*sin(Q) cos(Q)];
end
for i=n-1:-1:2
    F=F*a(:,:,i);
end
i=1;
a1=zeros(2,1);
a1(1,1)=a(1,1,i);
a1(2,1)=a(2,1,i);
F=F*a1;
y=real(F);
end

