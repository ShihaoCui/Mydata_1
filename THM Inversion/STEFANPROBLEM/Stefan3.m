% *************************************************************************
% *****                                                               *****
% *****                  Two phase Stefan problem                     *****
% *****              one-dimenisonal Finite Element Code              *****
% *****                         1D Models                             *****
% *****                      by Emad Norouzi                          *****
% *****                         June 2024                             *****
% *****                                                               *****
% *************************************************************************
clear
close all
clc
tic

% *************************************************************************
%            INPUT
% *************************************************************************

% Time
duration = 240*60*60          ; % [s]
dt       = 360                 ; % [s]
nstep    = round(duration/dt) ;
Time     = zeros(1,nstep+1)   ;

global node element

% General properties
poro    = 1.0       ; % Initial porosity
rho_s   = 2000      ; % Density of soil  [kg/m3]
rho_w   = 1000      ; % Density of water [kg/m3]
rho_i   = 920       ; % Density of ice   [kg/m3]
gravity = 9.81      ; % Gravity [m/s2]

% Thermal properties
Tr      = 273.15 ; % Reference temperature           [K]
k_parameter = 5.0 ;
landa_s = 1.1    ; % Thermal conductivity of soil    [J/s/m/K]
landa_w = 0.58   ; % Thermal conductivity of water   [J/s/m/K]
landa_i = 2.20   ; % Thermal conductivity of ice     [J/s/m/K]
cap_s   = 900    ; % Specific heat capacity of soil  [J/kg/K]
cap_w   = 4190   ; % Specific heat capacity of water [J/kg/K]
cap_i   = 2090   ; % Specific heat capacity of ice   [J/kg/K]
Lf      = 3.34e5 ; % Latent heat of fusion           [J/kg]

% Element Properties
elemType1   = 'L2'           ;
integ       = 'GAUSS'	     ;
order1      =  2             ;

% ***********************************************************************
%        MESHING and GEOMETRY
% ***********************************************************************
% Parameters
bar_length = 4;       % Length of the bar in meters
dx = 0.005;           % Node spacing in meters

% Number of nodes
numnode = bar_length / dx + 1;

% Generate node coordinates
node = linspace(0, bar_length, numnode)';

% Generate element connectivity
numelem = numnode - 1;
element = zeros(numelem, 2);
for i = 1:numelem
    element(i, :) = [i, i + 1];
end

FixedNodesT  = [1; 801];


% *************************************************************************
%        DEFINE VARIABLES
% *************************************************************************

% Number of unknowns
total_unknown = numnode ;
temp_unknown  = numnode ;

% Initial conditions

% Unknown
T  = zeros(1*numnode,1)+263.15   ;
T(1) = 308.15;

% Degree Of freedom
Tdofs  = FixedNodesT      ;

% Number of dofs
numTdof   = numnode ;

% *************************************************************************
%        INITIAL CONDITIONS
% *************************************************************************

% Unknowns
Xn    = T ;

tol_r = 1e-6 ;
[W1,Q1] = quadrature(order1,'GAUSS',1);


for step=0:1:nstep-1
    
    converged  = false      ;
    iter       = 0          ;
    
    % Stiffness matrix
    Ktt = sparse(numnode,numnode) ;
    Ctt = sparse(numnode,numnode) ;
    
    % Forces
    fT  = zeros(temp_unknown,1) ;
        
    % Finial unknowns
    X_new = Xn      ;
    T     = X_new   ;
    
    for iel = 1 : numelem
        
        sctrT = element(iel,:);
        
        for jel = 1 : size(W1,1)
            pt = Q1(jel,:);       % quadrature point
            % Bfem and Nfem for Q4
            sctrT = element(iel,:);
            nn   = length(sctrT);
            [N,dNdxi] = lagrange_basis(elemType1,pt); % element shape functions
            J0   = node(sctrT,:)'*dNdxi;              % element Jacobian matrix
            dNdx = dNdxi/J0;                          % derivatives of N w.r.t XY

            Nt= N';
            Bt= zeros(1,nn);
            Bt(1,1:1:nn)  = dNdx(:,1)' ;

            T_temp  = Nt*T(sctrT);

            % ---------------------
            % Auxiliary relations
            % ---------------------
%             if T_temp >= Tr
%                 si = 0;
%                 sw = 1-si;
%                 dsidT = 0;
%             else
                X = exp(k_parameter*(T_temp-Tr));
                si = 1/(1+X);
                dsidT = -k_parameter*X/(1+X)/(1+X);
                sw = 1-si;
%             end
                
            landa    = (1-poro)*landa_s+(poro*sw)*landa_w+(poro*si)*landa_i;
            rhoc_eff = (1-poro)*rho_s*cap_s+(poro*sw)*rho_w*cap_w+(poro*si)*rho_i*cap_i;
%             norm_T = norm(Bt*T(sctrT));
            
            % ---------------------
            % Assemble volume integrals
            % ---------------------
            Ktt(sctrT,sctrT)= Ktt(sctrT,sctrT)+ Bt'*landa*Bt*W1(jel)*det(J0);
            Ctt(sctrT,sctrT)= Ctt(sctrT,sctrT)+ Nt'*(rhoc_eff-Lf*rho_i*poro*dsidT)*Nt*W1(jel)*det(J0);
%             fT(sctrT,1)     = fT(sctrT,1)     + Nt'*(landa_i-landa_w)*poro*dsidT*norm_T^2*W1(jel)*det(J0);
            
        end % end of looping on GPs
    end
    clear iel jel

    % System of Jacoubian matrix:
    Jacoubian = Ctt/dt+Ktt ;
    
    C_matrix = Ctt ;
    K_matrix = Ktt ;
        
    % Vector of forces:
    Fext = fT;
        
    % Vector of residuals:
    residual = Fext - K_matrix*Xn ;
    
    % Essensial boundary condition
    bcwt = mean(diag(Jacoubian)); % a measure of the average size of an element in K
    
    % Temperature constraints
    Jacoubian(Tdofs,:) = 0 ;
    Jacoubian(:,Tdofs) = 0 ;
    Jacoubian(Tdofs,Tdofs) = bcwt*speye(length(Tdofs));
    
    residual(Tdofs)    = 0 ;

    % Newton-Raphson solution
    while (~ converged )
        iter = iter +1 ;
        
        % Solution of equations
        
        dX  = Jacoubian\residual ;
        X_new  = X_new+dX        ;
        
        % Vector of residuals:
        
        residual = Fext+C_matrix*Xn/dt-(K_matrix+C_matrix/dt)*X_new;
        residual(Tdofs)    = 0 ;

        disp ([ 'Norm In iter(' num2str(iter) ')=' num2str(norm(residual))]);
        if norm(residual) <tol_r
            converged=true;
            disp (' converged! ')
        end
        
    end
    
    % Final unknowns
    T  = X_new ;
    Xn = T     ;

    sec = floor(toc - floor(toc/60)*60)             ;
    mint= floor(floor(toc/60)-floor(toc/3600)*60)   ;
    hr  = floor(toc/3600)                           ;
    
    fprintf('\t\t\t %i Step',step+1)
    fprintf('\t\t\t %i Iterations',iter)
    fprintf('\t Elapsed time %02.0f:%02.0f:%02.0f\n',hr,mint,sec)
    fprintf('--------------------------------------------------------------\n')
    
    Te = stefan_analytical(step+1,dt);
    
    % Save data
    Time(step+2)= (step+1)*dt  ;
    
    if Time(step+2) == 4*60*60
        T4 = T;
        T4_analytical = Te;
    elseif Time(step+2) == 12*60*60
        T12 = T;
        T12_analytical = Te;
    elseif Time(step+2) == 28*60*60
        T28 = T;
        T28_analytical = Te;
    elseif Time(step+2) == 60*60*60
        T60 = T;
        T60_analytical = Te;
    elseif Time(step+2) == 124*60*60
        T124 = T;
        T124_analytical = Te;
    elseif Time(step+2) == 240*60*60
        T240 = T;
        T240_analytical = Te;
    end
end

figure (1)
hold on
plot(node,T4-273.15)
plot(node,T12-273.15)
plot(node,T28-273.15)
plot(node,T60-273.15)
plot(node,T124-273.15)
plot(node,T240-273.15)
grid on
xlim([0,1])

figure (2)
hold on
plot(node,T4_analytical)
plot(node,T12_analytical)
plot(node,T28_analytical)
plot(node,T60_analytical)
plot(node,T124_analytical)
plot(node,T240_analytical)
grid on
xlim([0,1])

save('T4','T4')
save('T12','T12')
save('T28','T28')
save('T60','T60')
save('T124','T124')
save('T240','T240')
