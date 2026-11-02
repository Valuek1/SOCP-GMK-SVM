function [accuracy, test_err, f1, gmean, train_err] = socp_svm( ...
    X_train, X_test, Train_label, Test_label, sigma_g, eta1, eta2)
%SOCP_SVM Fit the RBF-kernel SOCP-SVM baseline using SeDuMi.
% Inputs are scaled feature matrices and binary labels in {-1, +1}.

accuracy = NaN;
test_err = NaN;
f1 = NaN;
gmean = NaN;
train_err = NaN;

Train_label = Train_label(:);
Test_label = Test_label(:);
Train_label(Train_label ~= 1) = -1;
Test_label(Test_label ~= 1) = -1;
m = size(X_train, 1);

Kmat = exp(-sigma_g * pdist2(X_train, X_train).^2) + 1e-8 * eye(m);
Ktest = exp(-sigma_g * pdist2(X_test, X_train).^2);

idx1 = find(Train_label == 1);
idx2 = find(Train_label == -1);
n1 = numel(idx1);
n2 = numel(idx2);
if n1 == 0 || n2 == 0
    return;
end

mean1_vec = (1 / n1) * sum(Kmat(idx1, :), 1);
mean2_vec = (1 / n2) * sum(Kmat(idx2, :), 1);
K1 = Kmat(:, idx1);
K2 = Kmat(:, idx2);
H1 = eye(n1) - (1 / n1) * ones(n1);
H2 = eye(n2) - (1 / n2) * ones(n2);

try
    L_K = chol(Kmat, 'lower');
catch
    try
        L_K = chol(Kmat + 1e-6 * eye(m), 'lower');
    catch
        return;
    end
end

kappa1 = sqrt(eta1 / (1 - eta1));
kappa2 = sqrt(eta2 / (1 - eta2));

% SeDuMi variables: [b; lambda; t; s; u1; r1; u2; r2].
total_vars = 2 * m + n1 + n2 + 4;
eqs = m + n1 + n2 + 2;
i_b = 1;
i_lambda = 2:(m + 1);
i_t = m + 2;
i_s = (m + 3):(2 * m + 2);
i_u1 = 2 * m + 3;
i_r1 = (2 * m + 4):(2 * m + 3 + n1);
i_u2 = 2 * m + 4 + n1;
i_r2 = (2 * m + 5 + n1):(2 * m + 4 + n1 + n2);

A_eq = sparse(eqs, total_vars);
b_eq = zeros(eqs, 1);
A_eq(1:m, i_lambda) = -L_K';
A_eq(1:m, i_s) = speye(m);

A_eq(m + 1, i_lambda) = mean1_vec;
A_eq(m + 1, i_b) = 1;
A_eq(m + 1, i_u1) = -1;
b_eq(m + 1) = 1;

A_eq((m + 2):(m + 1 + n1), i_lambda) = -(kappa1 / sqrt(n1)) * H1 * K1';
A_eq((m + 2):(m + 1 + n1), i_r1) = speye(n1);

A_eq(m + n1 + 2, i_lambda) = -mean2_vec;
A_eq(m + n1 + 2, i_b) = -1;
A_eq(m + n1 + 2, i_u2) = -1;
b_eq(m + n1 + 2) = 1;

A_eq((m + n1 + 3):end, i_lambda) = -(kappa2 / sqrt(n2)) * H2 * K2';
A_eq((m + n1 + 3):end, i_r2) = speye(n2);

c_obj = sparse(total_vars, 1);
c_obj(i_t) = 1;
coneK.f = 1 + m;
coneK.q = [1 + m, 1 + n1, 1 + n2];

try
    [x, ~, info] = sedumi(A_eq', b_eq, c_obj, coneK, struct('fid', 0));
    if info.pinf ~= 0 || info.dinf ~= 0 || isempty(x)
        return;
    end
catch
    return;
end

b_svm = x(i_b);
lambda = x(i_lambda);
train_pred = sign(Kmat * lambda + b_svm);
train_err = mean(train_pred ~= Train_label);

test_pred = sign(Ktest * lambda + b_svm);
accuracy = mean(test_pred == Test_label);
test_err = mean(test_pred ~= Test_label);
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
