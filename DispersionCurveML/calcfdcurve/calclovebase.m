function pv= calclovebase(f,VS,H,den)
%   Summary of this function goes here.
%   pv= calclovebase(f,VS,H,VP,den)
%   Detailed explanation goes here.
%   The function is for calculating base mode of Love wave.
%
%   IN    f: row vector of frequency.
%        VS: row vector of shear wave velocity.
%         H: row vector of thickness of layer.
%       den: row vector of density.
%
%  OUT    pv: the phase velocity of base mode Love wave.
%
%  Example:
%  f=5:100;VS=[200 400];H=[10];den=[2000 2000];
%  pv=calclovebase(f,VS,H,den);
%
%  Author(s): Yan Yingwei
%  Copyright: 2017-2019 
%  Revision: 1.0  Date: 2/27/2017
%
%  Department of Geophysics, Jilin University.
if(nargin<=3)
    [~,c]=size(VS);
    den=2000*ones(1,c);
end
pvmulti = calclovemulti(f,VS,H,den);
pv=pvmulti(:,1)';
end

