function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
result = 0
sz = length(theta);

J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta.
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %


        theta_copy = theta;

        for iter1 = 1:sz

            result = 0;

            for i = 1:m

                result = result + ( X(i ,: ) * theta - y(i) ) * X(i,iter1);

            end

            result =  alpha * result / m;

            theta_copy(iter1) = theta_copy(iter1) -  result;

        end

        theta = theta_copy;

        % ============================================================

        % Save the cost J in every iteration
        J_history(iter) = computeCost(X, y, theta);

    end

end
