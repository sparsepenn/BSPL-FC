function [J, grad] = cost_DropoutNoise(theta, X, y, delta)

% Computes the cost of usingtheta as the parameter for regularized 
% logistic regression  with Dropout Noise
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
    (delta/(2*(1-delta)*m))*(((h_x.*(1-h_x)))'*(X.^2*theta.^2));

% Compute gradient 
grad(:,1)=((1/m)*((h_x-y)'*X))'...
          +(delta/(2*(1-delta)*m)) * (X'*((h_x.*(1-h_x).*(1-2*h_x)).*(X.^2*theta.^2)))...
          +(delta/(2*(1-delta)*m)) * ((bsxfun(@times,X,theta'))'*(h_x.*(1-h_x)));
