function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

arg = [0.01,0.03,0.1,0.3,1,3,10,30];

m = length(arg);

error = 1;

for i = 1:m
  for j = 1:m
    tmp_C = arg(i);
    tmp_sigma = arg(j);
    model= svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma)); 
    predictions = svmPredict(model, Xval);
    tmp_error = mean(double(predictions ~= yval));
    if error > tmp_error
      error = tmp_error;
      prop_C = tmp_C;
      prop_sigma = tmp_sigma;
    end
  end
end

C = prop_C;
sigma = prop_sigma;
      





% =========================================================================

end
