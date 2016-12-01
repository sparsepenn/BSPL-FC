%% Demo of BSPL-FC
% author: Yazhou Ren  Email: yazhou.ren@uestc.edu.cn
% In our experiments, we use Mark Schmidt's minFunc-implementation 
% (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
% of L-BFGS for optimization

Data='Iris';
load(Data);X=zscore(fea);X=NonZeroX(X);y=fixlabel(gnd); % preprocess
cp = cvpartition(y,'k',10);    % 10-fold cross validation
sigma = 0.1; % parameter of Gaussian noise
delta = 0.2; % parameter of Dropout noise

fprintf('dataset=%s, using SPL-GN\n',Data);
ClassFun = @(XTrain,yTrain,XTest)(myPredict(SPL_LR(XTrain,...
    double(yTrain), sigma, 'GaussianNoise', 'Unbalanced'), XTest));
cvMCR= crossval('mcr',X,y,'predfun',ClassFun,'partition',cp);
SPL_GN_precision = (1-cvMCR)*100

fprintf('dataset=%s, using BSPL-GN\n',Data);
ClassFun = @(XTrain,yTrain,XTest)(myPredict(SPL_LR(XTrain,...
    double(yTrain), sigma, 'GaussianNoise', 'Balanced'), XTest));
cvMCR= crossval('mcr',X,y,'predfun',ClassFun,'partition',cp);
BSPL_GN_precision = (1-cvMCR)*100

fprintf('dataset=%s, using SPL-DN\n',Data);
ClassFun = @(XTrain,yTrain,XTest)(myPredict(SPL_LR(XTrain,...
    double(yTrain), delta, 'DropoutNoise', 'Unbalanced'), XTest));
cvMCR= crossval('mcr',X,y,'predfun',ClassFun,'partition',cp);
SPL_DN_precision = (1-cvMCR)*100

fprintf('dataset=%s, using BSPL-DN\n',Data);
ClassFun = @(XTrain,yTrain,XTest)(myPredict(SPL_LR(XTrain,...
    double(yTrain), delta, 'DropoutNoise', 'Balanced'), XTest));
cvMCR= crossval('mcr',X,y,'predfun',ClassFun,'partition',cp);
BSPL_DN_precision = (1-cvMCR)*100