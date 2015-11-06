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

% x_i = X(1,:)';
% sigmoid(theta' * x_i)

%{estimates = y;
%or i = 1:m
%	h = sigmoid(theta' * X(i,:)');
%	estimates(i) = -y(i) * log(h) - (1 - y(i)) * log(1 - h);
%end
%J = 1/m * sum(estimates);
%
%for j = 1:size(grad,1)
%	estimates = y;
%	for i = 1:m
%		h = sigmoid(theta' * X(i,:)');
%		estimates(i) = (h - y(i)) * X(i,j);
%	end
%	grad(j) = 1/m * sum(estimates);
%end

left = -1 * (y .* log(sigmoid(X * theta)));
right = (1 - y) .* log(1 - sigmoid(X * theta));

J = (1/m) * sum(left - right);

grad = (X' * (sigmoid(X * theta) - y)) * (1/m);




% =============================================================

end
