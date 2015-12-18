# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.cm as cm

file_loc = '/home/abhishekb/ML_Project2/data/'

## Import data
import_test = sio.loadmat(file_loc + 'Test.mat')
import_train = sio.loadmat(file_loc + 'Train.mat')
X_train_raw = import_train['Xtrain']
X_test_raw = import_test['Xtest']
Y_train = import_train['Ytrain']

## Standardization
scaler = preprocessing.StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

## PCA and Feature Selection
pca = PCA(n_components=800)
selection = SelectKBest(k=450)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features.fit(X_train_scaled, Y_train.ravel())
print(pca.explained_variance_ratio_) 
X_train_reduced = combined_features.transform(X_train_scaled)
X_test_reduced = combined_features.transform(X_test_scaled)


## Create K folds
Y_kf = Y_train.ravel()
k_fold = StratifiedKFold(Y_kf, n_folds=10)

## Run cross validation
for train, test in k_fold:
    X1 = X_train_reduced[train]
    Y1 = Y_train[train]
    Y1 = Y1.ravel()
    
    X2 = X_train_reduced[test]
    Y2 = Y_train[test]    
    


    ## Train Classifiers on fold
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X1, Y1)
    #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=1.0).fit(X1, Y1)

    ## Score Classifiers on fold
    rbf_svc_score = rbf_svc.score(X2, Y2)
    lin_svc_score = lin_svc.score(X2, Y2)

    print "RBF: ",rbf_svc_score
    print "Linear: ", lin_svc_score

## Train final Classifiers
clf = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X_train_reduced, Y_train.ravel())
Y_predicted = clf.predict(X_test_reduced)

## Voting
Y_vote = np.zeros((len(Y_predicted),3))
for i in range (0,len(Y_predicted)):
    if (Y_predicted[i] == 0):
        Y_vote[i,0] = 1
    elif(Y_predicted[i] == 1):
        Y_vote[i,1] = 1
    elif(Y_predicted[i] == 3):
        Y_vote[i,2] = 1

## Save results to csv
np.savetxt('prediction.csv', Y_vote, fmt='%.1d',delimiter=',')