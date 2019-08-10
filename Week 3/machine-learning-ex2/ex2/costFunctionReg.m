function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);

%cost function 
h = sigmoid(X * theta); % don't understand. Formula states theta' * X NOT X * theta
summed = (1/m)*((-y'*log(h))-((1-y')*log(1-h)));
regularized = 0;
for i=2:n
  regularized += theta(i,1).^2;
endfor
J = summed + ((lambda/(2*m)) * regularized);

hx2 = sigmoid(X*theta);
thetaTemp = theta;
thetaTemp(1) = 0;

%calculate gradient(not gradient decent) ---------------------------
summation2 = [0,0,0];
for j=1:m,
 hx = sigmoid(theta * X(j,[1:3])); 
 summation2 += (hx- y(j, 1)).*X(j,[1:3]);
endfor
innnerSummation = (1/m)*summation2;

%grad = innnerSummation + ((lambda/m) * thetaTemp);

%Above does not work
%--------------------------------------------------------------------
grad = ((1 / m) * (hx2 - y)' * X) + lambda / m * thetaTemp'; 


% =============================================================

end
