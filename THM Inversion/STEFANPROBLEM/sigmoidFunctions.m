function [val, dval] = sigmoidFunctions(k, T_c, T)
    % Compute the sigmoid function value
    val = sigmoidValue(k, T_c, T);
    
    % Compute the first derivative of the sigmoid function
    dval = sigmoidFirstDerivative(k, T_c, T);
end

function val = sigmoidValue(k, T_c, T)
    x = k * (T - T_c);
    if x > 709.78
        val = 0;
    else
        val = 1 / (1 + exp(x));
    end
end

function dval = sigmoidFirstDerivative(k, T_c, T)
    f = sigmoidValue(k, T_c, T);
    if f^2 == 0
        dval = 0;
    else
        dval = -k * exp(k * (T - T_c)) * (f^2);
    end
end
