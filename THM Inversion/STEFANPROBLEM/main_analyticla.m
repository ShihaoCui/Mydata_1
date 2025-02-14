duration = 60*:1:240*60*60          ; % [s]
dt       = 360                 ; % [s]
step    = round(duration/dt) ;
Te = stefan_analytical(step,dt);

plot(step,Te);