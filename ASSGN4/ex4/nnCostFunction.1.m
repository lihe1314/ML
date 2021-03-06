function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

i = 0;
k = 0;
y_ = zeros(num_labels, m);

result = 0;

loop = 0;
temp = [];

for loop = 1:m
    temp = y_(:,loop);
    temp(y(loop)) = 1;
    y_(:,loop) = temp;
end

%this will be vector rised y

X = [ones(m, 1) X];

%h(x)

a2 = [];
a3 = [];
a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2, 1), 1) a2];
a3 = sigmoid(a2 * Theta2');
a4 = a3';
size(a3);

% %% using the loop
% for i = 1:m 

%     for k = 1:num_labels

%         result =  result - (y_(k,i) * log(a4(k,i)) + (1 - y_(k,i)) * log(1 - a4(k,i)));

%     end
% end

% J = result / m


% using the direct one line magic call
J = - sum(sum([y_  (1 - y_)] .* [log(a4) log(1 - a4)])) / m;


layers = 3; %This value is fixed

result = 0;

for j = 1:hidden_layer_size

    for k = 2:size(Theta1,2) % we skip the theta 0

        result = result + Theta1(j,k)^2;

    end

end

for j = 1:num_labels

    for k = 2:size(Theta2,2) %we skip the theta 0 

        result = result + Theta2(j,k)^2;

    end

end


result = lambda * result / (2 * m);

J = J + result;

% -------------------------------------------------------------

ab1 = zeros(1, size(X, 2));
ab2 = []; 
ab3 = [];


Delta1 = zeros(hidden_layer_size,size(X, 2)-1);
Delta2 = zeros(num_labels,hidden_layer_size);


% delta3 = (a3 - y_')';
% delta2 = ((Theta2(:,2:end)' * delta3)' .* sigmoidGradient(X * Theta1'))';

% Delta2 = Delta2 + delta3 * (a2(:,2:end));
% Delta1 = Delta1 + delta2 * (X(:,2:end));

% Theta1_grad = Delta1 / m;
% Theta2_grad = Delta2 / m;

for i = 1:m

    ab1 = X(i,2:end)';

    z2 = Theta1(:,2:end) * ab1;
    
    size(z2);

    ab2 = sigmoid(z2);

    z3 = Theta2(:,2:end) * ab2;

    size(z3);

    ab3 = sigmoid(z3);

    size(ab3);

    delta3 = ab3 - y_(:,i);

    size(delta3);

    delta2 = Theta2(:,2:end)' * delta3 .* ab2 .* (1 - ab2);

    size(delta2);

    delta1 = Theta1(:,2:end)' * delta2 .* ab1 .* (1 - ab1);

    size(delta1);

    Delta2 = Delta2 + delta3 * ab2';
    Delta1 = Delta1 + delta2 * ab1';
    
end

Theta1_grad = [ones(size(Delta1,1), 1) Delta1];
Theta2_grad = [ones(size(Delta2,1), 1) Delta2];

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
