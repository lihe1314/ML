function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

i = 1;

result = 0;

for i = 1:m

result = result + (-1 * y(i) * log( sigmoid( X(i,: ) * theta) )  - (1 - y(i)) * log(1 - sigmoid(X(i,: ) * theta)) );

end

J = result / m;

k = size(X,2);

j = 0;

theta_copy = theta;

for j = 1:k

  result = 0;

  for i = 1:m

    result = result + (sigmoid( X(i,: ) * theta ) - y(i)) * X(i, j);

  end

  theta_copy(j) = result / m;

end


grad = theta_copy;


% =============================================================

end
