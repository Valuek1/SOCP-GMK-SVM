function demo_socp_gmk()
%DEMO_SOCP_GMK Run the three model functions on a small synthetic dataset.

rng(0);
X = [0.35 * randn(24, 2) - 1; 0.35 * randn(24, 2) + 1];
Y = [-ones(24, 1); ones(24, 1)];
split = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(split), :);
X_test = X(test(split), :);
Y_train = Y(training(split));
Y_test = Y(test(split));

mu = mean(X_train);
scale = std(X_train);
scale(scale == 0) = 1;
X_test = (X_test - mu) ./ scale;
X_train = (X_train - mu) ./ scale;

tau = 0.5;
sigma_g = 0.2;
sigma_m = 20;
eta = 0.7;
C = 1;

[acc1, ~, f1_1, gm1, err1] = socp_gmk_svm( ...
    X_train, X_test, Y_train, Y_test, tau, sigma_g, sigma_m, eta, eta);
[acc2, ~, f1_2, gm2, err2] = socp_svm( ...
    X_train, X_test, Y_train, Y_test, sigma_g, eta, eta);
[acc3, ~, f1_3, gm3, err3] = svm_gmk( ...
    X_train, X_test, Y_train, Y_test, C, tau, sigma_g, sigma_m);

model = {'SOCP-GMK-SVM'; 'SOCP-SVM'; 'SVM-GMK'};
train_error = [err1; err2; err3];
accuracy = [acc1; acc2; acc3];
f1 = [f1_1; f1_2; f1_3];
gmean = [gm1; gm2; gm3];
disp(table(model, train_error, accuracy, f1, gmean));
end
