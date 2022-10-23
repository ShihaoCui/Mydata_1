function pv = calcmax(ff,VS,H,VP,den)
%   Summary of this function goes here.
%   pv= calcmax(f,VS,H,VP,den)
%   Detailed explanation goes here.
%   The function is for calculating maximal mode of Rayleigh wave.
%
%   IN   ff: row vector of frequency.
%        VS: row vector of shear wave velocity.
%         H: row vector of thickness of layer.
%        VP: row vector of primary wave velocity.
%       den: row vector of density.
%
%  OUT   pv: the phase velocity of maximal mode Rayleigh wave.
%
%  Example:
%  f=5:100;VS=[200 400];VP=[400 800];H=[10];den=[2000 2000];
%  pv=calcmax(f,VS,H,VP,den);
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
global mode_base n_mode
[l,nf]=size(ff);
ccc=calcmulti(ff,VS,H,VP,den);
ddk=0.00000001;
mode_base=[1,VS,H,VP,den];

for i=1:nf
   f=ff(i);
   mode_base(1)=f;
   CC=ccc(i,:);
   I=find(CC==0);
   CC(I)=[];
   [l,g]=size(CC);
   %gg(i)=g;
   uz=zeros(1,g);
   for j=1:g
      v=CC(j);
      [uz1,det_D]=engenuz(v);
      [uzd,det_Dd]=engenuz(1/(1/v-ddk/f));
      uz1=uz1/(det_D-det_Dd)*ddk;
      uz(j)=abs(uz1);
   end
   uz_max=max(uz);
   for j=1:g
      if uz(j)==uz_max
         ccc_cmp(i)=CC(j);
      end
   end
end
cd_fit=ccc_cmp;
pv=cd_fit;
end

