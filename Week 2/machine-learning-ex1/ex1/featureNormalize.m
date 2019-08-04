function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

temp1 = mean(X_norm,2);
mu(1,1) = temp1(1);
mu(1,2) = temp1(2);

for i=1:length(X_norm(1,:))
X_norm(i,1) = X_norm(i,1)- mu(1,1);
X_norm(i,2) = X_norm(i,2)- mu(1,2);
endfor

temp2 =  std(X_norm,1);
sigma(1,1) = temp2(1);
sigma(1,2) = temp2(2);

for j=1:length(X_norm(1,:))
X_norm(j,1) = X_norm(j,1)- sigma(1,1);
X_norm(j,2) = X_norm(j,2)- sigma(1,2);
endfor




% ============================================================

end
