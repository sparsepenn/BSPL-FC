function [J, grad] = cost_GaussianNoise(theta, X, y, sigma)

% Computes the cost of using theta as the parameter for regularized 
% logistic regression with Gaussian Noise
% and the gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta));

% Compute cost function
templog(:,1) = log(sigmoid(X*theta));
templog(:,2) = log(1-(sigmoid(X*theta)));
tempy(:,1) = y;
tempy(:,2) = 1-y;
temp = templog.*tempy;

h_x = sigmoid(X*theta);
% Formula for cost function. 
J = (1/m)*(-sum(temp(:,1))-sum(temp(:,2))) + ...
    (sigma^2/(2*m)) * norm(theta)^2 * sum(h_x.*(1-h_x));

% Compute gradient 
grad(:,1)=((1/m)*((h_x-y)'*X))'...
          +(sigma^2/(2*m))*(2*theta*sum(h_x.*(1-h_x)))...
          +(sigma^2/(2*m))*norm(theta)^2*((h_x.*(1-h_x).*(1-2*h_x))'*X)';
