function theta = SPL_LR(X, y, para_r, NoiseName, SPL_type)
% Self-paced learning with Logistic Regression
% returns the values of theta. Each row of theta corresponds
% to a single classifier for the number being considered.

% Some useful variables
[m,n] = size(X); % number of examples and features
numLabels = size(unique(y),1); % number of labels
theta = zeros(numLabels, n+1); % (n+1) to account for the x0 term
% initialTheta = zeros(n+1,1);

% Add ones to the X data matrix to account for x0
X = [ones(m, 1) X];

for i=1:numLabels
    yTemp = (y==i); % select all examples of particular number for training
    mfOptions.Method = 'lbfgs';
    %     mfOptions.optTol = 2e-2;
    %     mfOptions.progTol = 2e-6;
    mfOptions.LS = 2;
    mfOptions.LS_init = 2;
    mfOptions.MaxIter = 5;
    mfOptions.DerivativeCheck = 0;
    mfOptions.Display = 0;
    initialTheta = zeros(n+1,1);
    switch NoiseName
        case 'NoNoise'
            tempTheta = minFunc(@(t)(cost(t,X,yTemp,para_r)),initialTheta,mfOptions);
        case 'GaussianNoise'
            tempTheta = minFunc(@(t)(cost_GaussianNoise(t,X,yTemp,para_r)),initialTheta,mfOptions);
        case 'DropoutNoise'
            tempTheta = minFunc(@(t)(cost_DropoutNoise(t,X,yTemp,para_r)),initialTheta,mfOptions);
        otherwise
            error('Please check the Noise name!!!');
    end
    V = zeros(m,1);
    mfOptions.MaxIter = 200;
    for j=1:6
        % fprintf('SPL: the %d-th round\n', j);
        % Fix theta, optimize parameter V of self-paced learning
        V(:) = 0;
        switch SPL_type
            case 'Unbalanced'
                [~, ~, LossOfInstance] = myLRcost(tempTheta, X, yTemp, 0);
                sortLoss = sort(LossOfInstance);
                lambda_SPL = sortLoss(round(m*(0.4+0.1*j)));
                V(LossOfInstance<=lambda_SPL) = 1;
            case 'Balanced'
                [~, ~, LossOfInstance] = myLRcost(tempTheta, X, yTemp, 0);
                % sortLoss = sort(LossOfInstance);
                for k=1:numLabels
                    idx = find(y==k);
                    m_k = length(idx);
                    sortLoss_k = sort(LossOfInstance(idx));
                    lambda_SPL_k = sortLoss_k(round(m_k*(0.4+0.1*j)));
                    tmp = LossOfInstance<=lambda_SPL_k;
                    tmp(y~=k) = 0;
                    V(tmp) = 1;
                end
            otherwise
                error('Please check the SPL type!!!');
        end        
        % Fix V, optimize theta of self-paced learning
        switch NoiseName
            case 'NoNoise'
                tempTheta = minFunc(@(t)(cost(t,X(V==1,:),yTemp(V==1,:),para_r)),tempTheta,mfOptions);
            case 'GaussianNoise'
                tempTheta = minFunc(@(t)(cost_GaussianNoise(t,X(V==1,:),yTemp(V==1,:),para_r)),tempTheta,mfOptions);
            case 'DropoutNoise'
                tempTheta = minFunc(@(t)(cost_DropoutNoise(t,X(V==1,:),yTemp(V==1,:),para_r)),tempTheta,mfOptions);
            otherwise
                error('Please check the Noise name!!!');
        end
    end
    theta(i,:) = tempTheta';
    if(2==numLabels) % for binary classifiction, only need to compute once
        theta = theta(i,:);
        break;
    end
end