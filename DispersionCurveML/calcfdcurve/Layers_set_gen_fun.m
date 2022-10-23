function [H, VS, VP, den] = Layers_set_gen_fun()
    
    H=  [    5*(rand*0.2+0.9)                     10*(rand*0.2+0.9)                                                          ];
    VS= [   400*(rand*0.2+0.9)                    600*(rand*0.2+0.9)                 1000*(rand*0.2+0.9) ];
    VP= [   600*(rand*0.2+0.9)                   1000*(rand*0.2+0.9)                   2000*(rand*0.2+0.9) ];
    den=[  1000*(rand*0.2+0.9)                 1500*(rand*0.2+0.9)                 2000*(rand*0.2+0.9) ];
    
end