function  Vs= DPRFun(u1, u2,rho, freq)
    omega = 2*pi*freq;
    Vs = sqrt(2*(u1^2+omega.^2*u2^2)./(rho*(u1+sqrt(u1^2+omega.^2*u2^2))));
end