function pv= calcbase(f,VS,H,VP,den)
%   Summary of this function goes here.
%   pv= calcbase(f,VS,H,VP,den)
%   Detailed explanation goes here.
%   The function is for calculating base mode of Rayleigh wave.
%
%   IN    f: row vector of frequency.
%        VS: row vector of shear wave velocity.
%         H: row vector of thickness of layer.
%        VP: row vector of primary wave velocity.
%       den: row vector of density.
%
%  OUT    pv: the phase velocity of base mode Rayleigh wave.
%
%  Example:
%  f=5:100;VS=[200 400];VP=[400 800];H=[10];den=[2000 2000];
%  pv=calcbase(f,VS,H,VP,den);
%
%  Author(s): Yan Yingwei
%  Copyright: 2017-2019 
%  Revision: 1.0  Date: 2/27/2017
%
%  Department of Geophysics, Jilin University.
if(nargin<=4)
    [~,c]=size(VS);
    den=2000*ones(1,c);
end
if(nargin<=3)
    VP=2*VS;
end
global mode_base
[~,N]=size(f);
mode_base=[f(N),VS,H,VP,den];
pv=zeros(1,N);
pv(N)=fzero(@fastcalc,0.88*min(VS)); 
for i=N-1:-1:1
    mode_base(1)=f(i);
    pv(i)=fzero(@fastcalc,pv(i+1));
end
end

