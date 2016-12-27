function [] = q2()
    gradient_check();
end

function result = sigmoid(x)
    result = 1 / (1 + exp(-x));
end

function [y, z, P, J] = forwardprop(x, t, A, S, W)
    temp = A * [x;1];
    y = zeros(size(temp,1),1);
    for i = 1:size(temp,1)
        y(i) = sigmoid(temp(i));
    end
    
    temp = S * [y;1];
    z = zeros(size(temp,1),1);
    for i = 1:size(temp,1)
        z(i) = sigmoid(temp(i));
    end
    
    temp = W * [z;1];
    P = zeros(size(temp,1),1);
    s = 0;
    for i = 1:size(temp,1)
        P(i) = exp(temp(i));
        s = s + P(i);
    end
    P = P / s;
    J = -log(P(t));
end

function [grad_A, grad_S, grad_W] = backprop(x, t, A, S, W, N)
    [y, z, P, J] = forwardprop(x, t, A, S, W);
    I = zeros(N,1);
    I(t) = 1;
    
    grad_W = (P - I) * [z;1]';
    grad_S = ((W(:,1:30)' * (P - I)) .* (z .* (1.-z))) * [y;1]';
    grad_A = ((S(:,1:50)' * ((W(:,1:30)' * (P - I)) .* ...
        (z .* (1-z)))) .* y .* (1.-y)) * [x;1]';
end

function gradient_check()
    epsilon = 1e-4;
    M = 100;
    K = 50;
    D = 30;
    N = 10;
    nCheck = 1000;
    A = rand(K, M+1) * 0.1 - 0.05;
    S = rand(D, K+1) * 0.1 - 0.05;
    W = rand(N, D+1) * 0.1 - 0.05;
    x = rand(M,1) * 0.1 - 0.05;
    t = randsample(N, 1);
    
    [grad_A, grad_S, grad_W] = backprop(x,t,A,S,W, N);
    errA = [0];
    errS = [0];
    errW = [0];
    
    for i = 0:(nCheck-1)
        idx_x = randsample(K, 1);
        idx_y = randsample(M+1, 1);
        alter_A1 = A;
        alter_A2 = A;
        alter_A1(idx_x, idx_y) = alter_A1(idx_x, idx_y) + epsilon;
        alter_A2(idx_x, idx_y) = alter_A2(idx_x, idx_y) - epsilon;
        [y, z, P, J1] = forwardprop(x,t,alter_A1,S,W);
        [y, z, P, J2] = forwardprop(x,t,alter_A2,S,W);
        numerical_grad_A = (J1 - J2) / (2*epsilon);
        errA = [errA,abs(grad_A(idx_x, idx_y) - numerical_grad_A)];
        
        idx_x = randsample(D, 1);
        idx_y = randsample(K+1, 1);
        alter_S1 = S;
        alter_S2 = S;
        alter_S1(idx_x, idx_y) = alter_S1(idx_x, idx_y) + epsilon;
        alter_S2(idx_x, idx_y) = alter_S2(idx_x, idx_y) - epsilon;
        [y, z, P, J1] = forwardprop(x,t,A,alter_S1,W);
        [y, z, P, J2] = forwardprop(x,t,A,alter_S2,W);
        numerical_grad_S = (J1 - J2) / (2*epsilon);
        errS = [errS,abs(grad_S(idx_x, idx_y) - numerical_grad_S)];
        
        idx_x = randsample(N, 1);
        idx_y = randsample(D+1, 1);
        alter_W1 = W;
        alter_W2 = W;
        alter_W1(idx_x, idx_y) = alter_W1(idx_x, idx_y) + epsilon;
        alter_W2(idx_x, idx_y) = alter_W2(idx_x, idx_y) - epsilon;
        [y, z, P, J1] = forwardprop(x,t,A,S,alter_W1);
        [y, z, P, J2] = forwardprop(x,t,A,S,alter_W2);
        numerical_grad_W = (J1 - J2) / (2*epsilon);
        errW = [errW,abs(grad_W(idx_x, idx_y) - numerical_grad_W)];
    end
    % print out
    format long;
    mean(errA)
    mean(errS)
    mean(errW)
end