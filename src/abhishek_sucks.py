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
    1) Ridge alpha value has little to no effect on outcome
    2) Kernel Ridge regression 
        rbf, polynomial of higher degree, sigmoid kernels are garbage
        
    Results
        PCA 50 -  Ridge score of 63.5
        PCA 100 - Ridge score of 66.1 (rmse = .535688)
        PCA 200 - Ridge score of 64.2
              
        PCA 50 -  Lasso score of 62.5
        PCA 100 - Lasso score of 65
    
    
end Notes'''

file_loc = '/home/abhishekb/ML_Project2/data/'

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
x = import_train['x']
y = import_train['y']
z = import_train['z']
## Standardization
scaler = preprocessing.StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

## PCA and Feature Selection

'''pca = PCA(n_components=100)  
pca.fit(X_train_scaled)
#print(pca.explained_variance_ratio_) 
X_train_reduced = pca.transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)
'''

pca = PCA(n_components=800)
selection = SelectKBest(k=850)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features.fit(X_train_scaled, train_labels.ravel())
#print(pca.explained_variance_ratio_) 
X_train_reduced = combined_features.transform(X_train_scaled)
X_test_reduced = combined_features.transform(X_test_scaled)



## Train final Classifiers
#clf = Ridge(alpha=.5)
clf = Lasso(alpha=.03)
clf.fit(X_train_reduced, Y_train_raw)
Y_predicted = clf.predict(X_test_reduced)

## Save results to csv
np.savetxt('prediction.csv', Y_predicted, fmt='%.5f',delimiter=',')