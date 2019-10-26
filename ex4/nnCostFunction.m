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


%Part1
%Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). 
%This is most easily done using an eye() matrix of size num_labels, with 
%vectorized indexing by 'y'. A useful variable name would be "y_matrix", as this...

y_matrix = eye(num_labels)(y,:);


%perfom forward propogation. This code was taken from excersize 3
a1 = [ones(m, 1) X];
z2=a1*Theta1';
%Compute the sigmoid() of 'z2', then add a column of 1's, and it becomes 'a2'
a2 = sigmoid(z2);
m1=size(a2,1);
a2 = [ones(m1, 1) a2];
%Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
a3=sigmoid(a2*Theta2');
rcf=sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2));
J = sum(sum(-y_matrix.*log(a3) - (1-y_matrix).*log(1-a3)))/m +lambda*rcf/(2*m);

%back propogation
%m = the number of training examples 5000

%n = the number of training features, including the initial bias unit.401

%h = the number of units in the hidden layer - NOT including the bias unit 5

%r = the number of output classifications 10
%2: \delta_3? or d3 is the difference between a3 and the y_matrix. 
%The dimensions are the same as both, (m x r).
d3=a3-y_matrix;

%4: \delta_2? or d2 is tricky. It uses the (:,2:end) columns of Theta2.
% d2 is the product of d3 and Theta2 (without the first column), 
%then multiplied element-wise by the sigmoid gradient of z2. 
%The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2.
d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);
%5 Delta1 is the product of d2 and a1. 
%The size is (h x m)? (m x n) --> (h x n)
Delta1=d2'*a1;
%6 Delta2 is the product of d3 and a2. 
%The size is (r x m) \cdot? (m x [h+1]) --> (r x [h+1])
Delta2=d3'*a2;
% -------------------------------------------------------------

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad=Delta1/m+(lambda*Theta1)/m;
Theta2_grad=Delta2/m+(lambda*Theta2)/m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
