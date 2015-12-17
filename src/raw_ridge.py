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
#from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso

''' Notes
        Want to try without PCA or feature selection
        
    Results
        Ridge score 37
        Lasso score 63.6
    
    
end Notes'''

file_loc = '/home/abhishekb/ML_Project2/data/'

## Import data
import_test = sio.loadmat(file_loc + 'Test.mat')
import_train = sio.loadmat(file_loc + 'Train.mat')
import_providedidx = sio.loadmat(file_loc + 'provideIdx.mat')
import_missidx = sio.loadmat(file_loc + 'missIdx.mat')
import_providedata = sio.loadmat(file_loc + 'provideData_1000.mat')
import_events = sio.loadmat(file_loc + 'events_1000.mat')

provideIdx = import_providedidx['provideIdx'] - 1 # since Matlab indexing starts at 1
provideIdx = provideIdx.ravel()
missIdx = import_missidx['missIdx'] - 1 # since Matlab indexing starts at 1
missIdx = missIdx.ravel()
events = import_events['events']
X_test_raw = import_providedata['provideData'] # size = (1000, 3172)
XY_train_raw = import_train['Xtrain']
X_train_raw = XY_train_raw[:,provideIdx] # size = (500, 3172)
Y_train_raw = XY_train_raw[:,missIdx] # size = (500, 2731)

## Standardization
scaler = preprocessing.StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

## PCA and Feature Selection

X_train_reduced = X_train_scaled
X_test_reduced = X_test_scaled

## Create K folds
'''
k_fold = KFold(Y_train_raw.shape[0], n_folds=10)
for train, test in k_fold:
    X1 = X_train_reduced[train]
    Y1 = Y_train_raw[train]
    
    X2 = X_train_reduced[test]
    Y2 = Y_train_raw[test]    

    ## Train Classifiers on fold
    rdg_clf = Ridge(alpha=.5)
    rdg_clf.fit(X1,Y1)
    lso_clf = Lasso(alpha=.1)
    lso_clf.fit(X1,Y1)
        
    ## Score Classifiers on fold
    rdg_clf_score = rdg_clf.score(X2, Y2)
    lso_clf_score = lso_clf.score(X2, Y2)

    print "Ridge:  ", rdg_clf_score
    print "Lasso:  ", lso_clf_score

'''
## Train final Classifiers
clf = Ridge(alpha=.5)
clf.fit(X_train_reduced, Y_train_raw)
Y_predicted = clf.predict(X_test_reduced)

## Save results to csv
np.savetxt('prediction_ridge.csv', Y_predicted, fmt='%.5f',delimiter=',')

clf = Lasso(alpha=.1)
clf.fit(X_train_reduced, Y_train_raw)
Y_predicted = clf.predict(X_test_reduced)

## Save results to csv
np.savetxt('prediction_lasso.csv', Y_predicted, fmt='%.5f',delimiter=',')