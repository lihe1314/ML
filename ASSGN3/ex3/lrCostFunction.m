function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're fmany possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%get the 1's matix and get 0's matrix

% idx0 = 0;
% idx1 = 1;

% idx0 = ( y(:,1)==0 ); %get all the 0' index
% idx1 = ( y(:,1)==1 ); %get all the 1' index
% ny_t0 = y(idx0,:);
% nX_t0 = X(idx0,:);
% ny_t1 = y(idx1,:);
% nX_t1 = X(idx1,:);

% ny_t0;
% ny_t1;
% nX_t0;
% nX_t1;

% len1 = length(ny_t1);
% len0 = length(ny_t0);

% result1 = sum( log(sigmoid(nX_t1 * theta)))
% result0 = sum( log(1 - sigmoid(nX_t0 * theta)))

% J = -1 * (result1 + result0) / m + sum(theta(2:end) .^ 2) * lambda / (2 * m)

expx = [[log(sigmoid(X * theta))] ; [log(1 - sigmoid(X * theta))]];
expy = [y ; 1-y];

J = -1 * sum(expx .* expy) / m + sum(theta(2:end) .^ 2) * lambda / (2 * m);


%mark the first theta vector to be 0 as not need to be considered
k_ = theta;
k_(1) = 0;

grad = X' * (sigmoid(X * theta) .- y) / m + k_ * lambda / m;

% =============================================================



end
