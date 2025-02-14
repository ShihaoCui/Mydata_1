function Te = stefan_analytical(step,dt)


% Liquid phase properties
rho_w = 1000;        % Density of liquid (kg/m^3)
cp_w = 4190;         % Specific heat capacity of liquid (J/kgK)
landa_w = 0.58;    % Thermal conductivity of liquid (W/mK)

% Ice phase properties
rho_i = 920;         % Density of ice (kg/m^3)
cp_i = 2090;         % Specific heat capacity of ice (J/kgK)
landa_i = 2.2;     % Thermal conductivity of ice (W/mK)

% Temperatures
Ti = 263.15;         % Ice phase temperature (K)
T_r = 273.15;         % Melting temperature (K)

% Derived parameters
alfa_w = landa_w / (rho_w * cp_w); % Thermal diffusivity of liquid
alfa_i = landa_i / (rho_i * cp_i); % Thermal diffusivity of ice

rho_star = rho_w / rho_i; % Dimensionless parameter
alpha_star = sqrt(alfa_w / alfa_i); % Dimensionless parameter

% Solve the transcendental equation for Lambda
LAMBDA = 0.3933292421;


t_steps= (step+1+1000)*dt;

Te = Ti + (T_r - Ti) * erfc(4 / (2 * sqrt(alfa_i * t_steps))-(1-rho_star)*alpha_star*LAMBDA) / erfc(rho_star * alpha_star * LAMBDA);

end