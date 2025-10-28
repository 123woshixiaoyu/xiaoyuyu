#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import neighbors

if __name__ == "__main__":
    # load data
    datadir = './'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()  # shape (200, 361)
    Y = data['Y'].transpose().ravel()  # shape (200,)
    nbr_examples = np.size(Y, 0)

    nbr_trials = 100
    err_rates_test = np.zeros((nbr_trials, 3))
    err_rates_train = np.zeros((nbr_trials, 3))

    for i in range(nbr_trials):
        # 80/20 split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=None
        )
        nbr_train_examples = np.size(Y_train, 0)
        nbr_test_examples = np.size(Y_test, 0)

        # Decision Tree
        tree_model = tree.DecisionTreeClassifier()
        tree_model.fit(X_train, Y_train)

        # Linear SVM
        svm_model = svm.SVC(kernel="linear")  # linear kernel
        svm_model.fit(X_train, Y_train)

        # kNN
        nn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
        nn_model.fit(X_train, Y_train)

        # --- Test predictions ---
        predictions_test_tree = tree_model.predict(X_test)
        predictions_test_svm = svm_model.predict(X_test)
        predictions_test_nn = nn_model.predict(X_test)

        # --- Train predictions ---
        predictions_train_tree = tree_model.predict(X_train)
        predictions_train_svm = svm_model.predict(X_train)
        predictions_train_nn = nn_model.predict(X_train)

        # --- Compute error rates ---
        err_rate_test_tree = np.count_nonzero(predictions_test_tree - Y_test) / nbr_test_examples
        err_rate_test_svm = np.count_nonzero(predictions_test_svm - Y_test) / nbr_test_examples
        err_rate_test_nn = np.count_nonzero(predictions_test_nn - Y_test) / nbr_test_examples

        err_rate_train_tree = np.count_nonzero(predictions_train_tree - Y_train) / nbr_train_examples
        err_rate_train_svm = np.count_nonzero(predictions_train_svm - Y_train) / nbr_train_examples
        err_rate_train_nn = np.count_nonzero(predictions_train_nn - Y_train) / nbr_train_examples

        # Store
        err_rates_test[i, 0] = err_rate_test_tree
        err_rates_test[i, 1] = err_rate_test_svm
        err_rates_test[i, 2] = err_rate_test_nn

        err_rates_train[i, 0] = err_rate_train_tree
        err_rates_train[i, 1] = err_rate_train_svm
        err_rates_train[i, 2] = err_rate_train_nn

    # 输出平均错误率
    print("Mean test error rates (tree, svm, knn):")
    print(np.mean(err_rates_test, axis=0))
    print("Mean train error rates (tree, svm, knn):")
    print(np.mean(err_rates_train, axis=0))




