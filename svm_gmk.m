function [accuracy, test_err, f1, gmean, train_err] = svm_gmk( ...
    X_train, X_test, Train_label, Test_label, C, tau, sigma_g, sigma_m)
%SVM_GMK Fit the GMK-SVM baseline using LIBSVM and a precomputed kernel.
% Inputs are scaled feature matrices and binary labels in {-1, +1}.

Train_label = Train_label(:);
Test_label = Test_label(:);

K_train = gmk_kernel(X_train, X_train, tau, sigma_g, sigma_m);
K_test = gmk_kernel(X_test, X_train, tau, sigma_g, sigma_m);
train_index = (1:size(K_train, 1))';
test_index = (1:size(K_test, 1))';

options = sprintf('-t 4 -c %.17g -m 512 -h 0 -q', C);
model = svmtrain(Train_label, [train_index, K_train], options);
if isempty(model)
    error('LIBSVM could not fit the SVM-GMK model.');
end

train_pred = svmpredict(Train_label, [train_index, K_train], model, '-q');
test_pred = svmpredict(Test_label, [test_index, K_test], model, '-q');
train_err = mean(train_pred ~= Train_label);
test_err = mean(test_pred ~= Test_label);
accuracy = mean(test_pred == Test_label);

tp = sum((test_pred == 1) & (Test_label == 1));
fp = sum((test_pred == 1) & (Test_label == -1));
fn = sum((test_pred == -1) & (Test_label == 1));
tn = sum((test_pred == -1) & (Test_label == -1));
precision = tp / (tp + fp + eps);
recall = tp / (tp + fn + eps);
f1 = 2 * precision * recall / (precision + recall + eps);
specificity = tn / (tn + fp + eps);
gmean = sqrt(recall * specificity);
end

function K = gmk_kernel(X1, X2, tau, sigma_g, sigma_m)
dist2 = pdist2(X1, X2).^2;
K = (1 - tau) * exp(-sigma_g * dist2) + tau * exp(-sigma_m * dist2);
end
