function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    runningTotal1 = 0;
    runningTotal2 = 0;
    for i=1:m,
      hypothesis = theta(1,1) + (theta(2,1)*X(i,2));
      summation1 = sum(hypothesis .- y(i,1)); %for first summation dont need to multiply by X as per formula
      summation2 = (sum(hypothesis .- y(i,1)))* X(i, 2); %1x2 matrix 
      runningTotal1 += summation1;
      runningTotal2 += summation2;
    endfor
  
    interum = 1/m;
    t1 = interum * runningTotal1;
    t2 = interum * runningTotal2;

    theta(1) = theta(1) - alpha*t1;
    theta(2) = theta(2) - alpha*t2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
