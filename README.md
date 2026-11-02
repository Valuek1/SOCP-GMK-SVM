# SOCP-GMK-SVM: core MATLAB code

This folder contains the model-level implementation of SOCP-GMK-SVM and two comparison models. It does not include dataset files, hyperparameter search, plotting code, or the full experimental pipeline used to produce the manuscript tables.

## Files

| File | Purpose |
| --- | --- |
| `socp_gmk_svm.m` | SOCP-GMK-SVM with a dual-scale Gaussian memory kernel |
| `socp_svm.m` | SOCP-SVM baseline with a Gaussian RBF kernel |
| `svm_gmk.m` | SVM-GMK baseline with a dual-scale Gaussian memory kernel |
| `demo_socp_gmk.m` | Small synthetic-data example calling all three models |

## Requirements

- MATLAB with Statistics and Machine Learning Toolbox (`pdist2`, `cvpartition`).
- [SeDuMi](https://github.com/sqlp/sedumi) on the MATLAB path for the two SOCP models.
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) MATLAB interface on the MATLAB path for SVM-GMK. Check that `which svmtrain` and `which svmpredict` resolve to LIBSVM.

SeDuMi and LIBSVM are third-party dependencies and are not included here.

## Quick start

Add SeDuMi and LIBSVM to the MATLAB path, change to this folder, and run:

```matlab
demo_socp_gmk
```

For your own data, call a model function with numeric feature matrices (`rows = samples`, `columns = features`) and binary labels encoded as `+1` and `-1`. Standardize the test features using **training-set** means and standard deviations before calling the functions. The functions return `[accuracy, test_error, f1, gmean, training_error]`.

```matlab
[accuracy, test_error, f1, gmean, training_error] = socp_gmk_svm( ...
    X_train, X_test, y_train, y_test, tau, sigma_g, sigma_m, eta1, eta2);
```

`tau` is the memory-kernel weight; `sigma_g` and `sigma_m` are the generalization and memory Gaussian kernel parameters, respectively (`sigma_m > sigma_g > 0`); and `eta1`, `eta2` are the class probability bounds in `(0, 1)`. `svm_gmk.m` additionally takes the LIBSVM penalty parameter `C` before `tau`.

The demo uses fixed parameters solely to illustrate the function calls. Numerical reproduction of a manuscript table also requires its dataset preparation, split sequence, and hyperparameter selection procedure.
