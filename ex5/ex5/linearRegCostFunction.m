function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hyp = X*theta;
theta_reg = [0;theta(2:end, :);]; % Regularization is applied from second element onwards not theta_0, so Theta_0 is set to zero
%J = (1/(2*m))*sum((hyp-y).^2) + (lambda/(2*m))*sum(theta.^2); % j starts from 1 not zero for regularization
J = (1/(2*m))*sum((hyp-y).^2) + (lambda/(2*m))*sum(theta_reg.^2);


% Gradient Values

grad = (1/m) * X' * (hyp - y) + (lambda/m) * theta_reg;
%grad_1 =  (1/m)*sum((hyp-y).*X(:,(2:end)));
% grad(2) =  (1/m)*sum((hyp-y).*X(:,(2:end))) + (lambda/m)*(theta(2:end));
% =========================================================================

grad = grad(:);

end
