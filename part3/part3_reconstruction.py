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
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm

#file_loc = '/home/abhishekb/ML_Project2/data/'
file_loc = '/home/apark/Homework/ML_Project2/data/'
## Import data
import_test = sio.loadmat(file_loc + 'Test.mat')
import_train = sio.loadmat(file_loc + 'Train.mat')
import_providedidx = sio.loadmat(file_loc + 'provideIdx.mat')
import_missidx = sio.loadmat(file_loc + 'missIdx.mat')
import_providedata = sio.loadmat(file_loc + 'provideData_1000.mat')
import_events = sio.loadmat(file_loc + 'events_1000.mat')
import_test_labels = sio.loadmat(file_loc + 'Ytest.mat')

provideIdx = import_providedidx['provideIdx'] - 1 # since Matlab indexing starts at 1
provideIdx = provideIdx.ravel()
missIdx = import_missidx['missIdx'] - 1 # since Matlab indexing starts at 1
missIdx = missIdx.ravel()
events = import_events['events']
X_test_raw_1 = import_test['Xtest']
X_test_raw = import_providedata['provideData'] # size = (1000, 3172)
XY_train_raw = import_train['Xtrain']
X_train_raw_2 = XY_train_raw[:,provideIdx] # size = (500, 3172)
X_train_raw_1 = X_test_raw_1[:,provideIdx]  
Y_train_raw_2 = XY_train_raw[:,missIdx] # size = (500, 2731)
Y_train_raw_1 = X_test_raw_1[:,missIdx]
X_train_raw = np.vstack((X_train_raw_1, X_train_raw_2))
Y_train_raw = np.vstack((Y_train_raw_1, Y_train_raw_2))
train_labels = np.vstack((import_test_labels['Ytest'],import_train['Ytrain'])) #labels of the original train data

## Approach 2
## Standardization
scaler = preprocessing.StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)

## PCA and Feature Selection
pca = PCA(n_components=800)
selection = SelectKBest(k=850)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features.fit(X_train_scaled, train_labels.ravel())
X_train_reduced = combined_features.transform(X_train_scaled)

clf = MultiTaskLasso(alpha=.3)
clf.fit(X_train_reduced, Y_train_raw)
Y_predicted = clf.predict(X_train_reduced)


    #Y_predicted = np.vstack(())
    replacement_mod = iteration; # 1500-perfect 0-predicted
    print 'replacement_mod', replacement_mod
    Y_reconstruction = Y_train_raw
    for i in range(0,Y_train_raw.shape[0]):
        if (np.mod(i,replacement_mod) == 0):
            print i
            Y_reconstruction[i,:] = Y_predicted[i,:] 
            
    recon_acc = mean_squared_error(Y_train_raw, Y_reconstruction)
    print 'reconstruction accuracy:', recon_acc
    
    
    X_train_raw_new = np.hstack((X_train_raw,Y_predicted)) 
    
    ## Standardization
    scaler = preprocessing.StandardScaler().fit(X_train_raw_new)
    X_train_scaled_new = scaler.transform(X_train_raw_new)
    
    ## PCA and Feature Selection
    pca = PCA(n_components=800)
    selection = SelectKBest(k=450)
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    combined_features.fit(X_train_scaled_new, train_labels.ravel())
    X_train_reduced_new = combined_features.transform(X_train_scaled_new)
    
    ##Approach 1
    ## Standardization
    scaler = preprocessing.StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    
    ## PCA and Feature Selection
    pca = PCA(n_components=800)
    selection = SelectKBest(k=450)
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    combined_features.fit(X_train_scaled, train_labels.ravel())
    X_train_reduced = combined_features.transform(X_train_scaled)
    
    ## Create K folds
    Y_kf = train_labels.ravel()
    k_fold = StratifiedKFold(Y_kf, n_folds=5)
    
    ## Run cross validation
    
    sum1 = 0
    sum2 = 0
    for train, test in k_fold:
        X1 = X_train_reduced[train]
        Z1 = train_labels[train]
        Z1 = Z1.ravel()
        
        X2 = X_train_reduced[test]
        Z2 = train_labels[test]
        
        ## Train Classifiers on fold
        clf_1 = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X1, Z1)
        
        X1_new = X_train_reduced_new[train]
        
        X2_new = X_train_reduced_new[test]
    
        ## Train Classifiers on fold
        clf_2 = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X1_new, Z1)
            
        ## Score Classifiers on fold
        clf_1_score = clf_1.score(X2, Z2)
        clf_2_score = clf_2.score(X2_new, Z2)
        
       # print "Approach 1 ",clf_1_score
       # print "Approach2 ",clf_2_score
        sum1 = sum1 + clf_1_score
        sum2 = sum2 + clf_2_score
    
    print 'final result'
    print 'Approach 1 :', sum1/5.0
    print 'Approach 2 :', sum2/5.0
        
    ## Train final Classifiers
    #clf = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X_train_reduced, Y_train.ravel())
    #Y_predicted = clf.predict(X_test_reduced)
    
    ## Save results to csv
    #np.savetxt('prediction.csv', Y_predicted, fmt='%.1d',delimiter=',')