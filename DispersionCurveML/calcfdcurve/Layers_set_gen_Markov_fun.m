function [H, VS, VP, den] = Layers_set_gen_Markov_fun()
% clear
% clc

   L = 10;
   %% the H
   H = ones(1,L);
   VS = ones(1,L+1);
   VP = ones(1,L+1);
    den = ones(1,L+1);
   H(1) = 2;
   for i=2:1:L
       H(i) =1.10*H(i-1); 
   end
   clear i;
   
   %% The Vs
   % the half  space:Vs\
   VS_min = 150;
   VS_max = 350;
   
   VS(end) = 1500;
   VS(1) = rand*(VS_max -VS_min )+VS_min;
   lbd_min = 0.02;
   lbd_max = 0.3;
   
   tau = 0.9;
   alpha= rand;
   % case 1
   if alpha <= tau
        for j=2:1:L
            lbd = rand*(lbd_max-lbd_min)+lbd_min;
            VS(j) = VS(j-1)+lbd*VS(j-1);
            if VS(j) >=1200
                VS(j) = 1200;
            end
        end
    clear j;
    
    % case 2
   elseif (tau < alpha) & (alpha <= (tau+(1-tau)/2))
        for j=2:1:L
            aa = VS(j-1)+lbd_max*VS(j-1);
            bb= min(VS(j-1)+300,1200);
            VS(j) = rand*(bb-aa)+aa;
        end
        clear j;
        
    % case 3
   elseif  ((tau+(1-tau)/2) <= alpha) & (alpha <1.0)
        for j=2:1:L
            aa1 = max(VS(j-1)-300,150);
            bb1 = VS(j-1)+lbd_min*VS(j-1);
            VS(j) = rand*(bb1-aa1)+aa1;
         end
        
   end
  
   %% The Vp
   H_k = ones(1,L);
   for j=1:1:L
       H_k(j) = sum(H(1:j)); 
   end
   VP(1:L) = VS(1:L)./(0.5684*(5*10^(-3)*H_k).^0.163);
   VP(L+1) = VS(L+1)/(0.5684*(5*10^(-3)*H_k(L))^0.163);
   
%     plot(VS);
% hold on
% plot(VP)

%% The den

den = 1.2475+0.399*(VP.*10^-3)-0.026*(VP*10^-3).^2;
den = 10^3*den;


den = den;
VS = VS;
VP = VP;
H = H;

end