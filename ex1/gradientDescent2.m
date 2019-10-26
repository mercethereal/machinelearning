function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
thetaZero=0;
thetaOne=0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	s1=0;
	s2=0;
	for i=1:m
		s1=s1+(X(i,:)*theta-y(i));
		s2=s2+(X(i,:)*theta-y(i))*X(i,2);
	end
    
	thetaZero=thetaZero-alpha*s1/m;
	thetaOne=thetaOne-alpha*s2/m;
	
	theta=[thetaZero;thetaOne];


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
