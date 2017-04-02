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

% first get the #number of features from the X
sz = size(X,2);


%%loop the features, get the mean and update the each feature value, save the mean for each feature
for i=1:sz


  mu(i) = mean(X_norm( : , i ));

  X_norm(: , i) = X_norm(: , i) - mu(i);

end


%do the std for each feature
for i=1:sz

  sigma(i) = std(X_norm( : , i ));

  X_norm(: , i) = X_norm(: , i) / sigma(i);

end

% ============================================================

end
