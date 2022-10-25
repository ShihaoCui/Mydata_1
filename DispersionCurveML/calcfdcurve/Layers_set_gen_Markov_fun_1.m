function [H, VS, VP, den] = Layers_set_gen_Markov_fun_1()
% clear
% clc

   L = 10;
   %% the H
   H = ones(1,L);
   VS = ones(1,L+1);
   VP = ones(1,L+1);
   den = ones(1,L+1);
   
   H(1) = 3;
   for i=2:1:L
       H(i) =1.0*H(i-1); 
   end
   clear i;
   
   %% The Vs
   % the half  space:Vs
   VS1_min = 150;
   VS1_max = 250;
   VS_end_max = 600;
   
   VS(end) = VS_end_max;
   
   
   VS(1) = rand*(VS1_max -VS1_min )+VS1_min;
   
   
   lbd_min = 0.01;
   lbd_max = 0.2;
   
   tau = 0.8;
   alpha= rand;
   
   
   % case 1
   if alpha <= tau
%    if tau == 1
        for j=2:1:L
            lbd = rand*(lbd_max-lbd_min)+lbd_min;
            VS(j) = VS(j-1)+lbd*VS(j-1);
            if VS(j) >=VS_end_max
                VS(j) = VS_end_max;
            end
        end
    clear j;
    

      % tau = 4   
      elseif alpha > tau
%    elseif tau ==4
       prob4 = 0.5;
       lbd_decr_max = 0.3;
       lbd_decr_min = 0.01;
        for j=2:1:L
            Tau4 = rand;
            if Tau4<=prob4
                lbd = rand*(lbd_decr_max-lbd_decr_min)+lbd_decr_min;
                VS(j) = VS(j-1)+lbd*VS(j-1);
            
            else
               lbd_decr = rand*(lbd_decr_max - lbd_decr_min)+lbd_decr_min;
                VS(j) = VS(j-1)-lbd_decr*VS(j-1);
            end
            
            if VS(j) >=VS_end_max
               VS(j) = VS_end_max;
            end
            
            if VS(j)<200
               VS(j)=200;
            end    
            
        end
  
     % case 2
%    elseif (tau < alpha) & (alpha <= (tau+(1-tau)/2))
%    elseif tau == 2
%         for j=2:1:L
%             aa = min(VS(j-1)+lbd_max*VS(j-1),VS_end_max)
%             bb= VS(j-1)+50
%             VS(j) = rand*(bb-aa)+aa;
%             if VS(j) > VS_end_max
%                 VS(j) = VS_end_max;
%             end
%         end
%         clear j;
        
%     case 3
% %    elseif  ((tau+(1-tau)/2) <= alpha) & (alpha <1.0)
%    elseif tau ==3
%         for j=2:1:L
%             aa1 = max(VS(j-1)-50,250);
%             bb1 = VS(j-1)+0.1*VS(j-1);
%             VS(j) = rand*(aa1-bb1)+bb1;
%         end
   end
  
   %% The Vp
   H_k = ones(1,L);
   
   for j=1:1:L
       H_k(j) = sum(H(1:j)); 
   end
%    VP(1:L) = VS(1:L)./(0.5684*(5*10^(-3)*H_k).^0.163);
%    VP(L+1) = VS(L+1)/(0.5684*(5*10^(-3)*H_k(L))^0.163);
%    

VP = 2*VS;
% hold on
% plot(VP)

%% The den

% den = 1.2475+0.399*(VP.*10^-3)-0.026*(VP*10^-3).^2;
% den = 10^3*den;

den = ones(1,L+1);
den = 1800*den;


den = den;
VS = VS;
VP = VP;
H = H;
% VS
% stairs([0,H_k],VS);
end