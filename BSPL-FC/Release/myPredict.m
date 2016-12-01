function prediction = myPredict(theta, X)

% This function returns the predicted label for the given X based on the
% highest probablity compared among each of the classifiers.

m = size(X, 1);


% Add ones to the X data matrix to account for x0
X = [ones(m, 1) X];

if(size(theta,1)>2) % multi-class
    %     prediction = zeros(m, 1);
    tempProb = X * theta';
    [output,prediction] = max(tempProb,[],2);
else
    prediction = 2+zeros(m,1);
    temp = X * theta';
    prediction(temp>0,1) = 1;
end