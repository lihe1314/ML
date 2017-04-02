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



i = 1;

k = size(X,2);

result = 0;
result1 = 0

for i = 1:m

    result = result + (-1 * y(i) * log( sigmoid( X(i,: ) * theta) )  - (1 - y(i)) * log(1 - sigmoid(X(i,: ) * theta)) );

end

result = result / m;

for j=2:k

    result1 = result1 +  theta(j)^2;

end

result1 = lambda * result1 / (2 * m);

result = result + result1;

J = result;


j = 0;
extra = 0;
theta_copy = theta; %keep the copy as we need to update all theta at the same time

%loop through the size of the theta
for j = 1:k

  %reset the result
  result = 0;

  for i = 1:m

    result = result + (sigmoid( X(i,: ) * theta ) - y(i)) * X(i, j) ;

  end

  result = result / m;

  if(j>1)
  
      extra = lambda * theta(j)/m;
  
  else
  
      extra = 0;

  endif

  
  result = result + extra;

  theta_copy(j) = result;

end


%the current iteration the theta_copy was all updated theta value assign back to grad
grad = theta_copy;

% =============================================================

end
