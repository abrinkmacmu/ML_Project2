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
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso

''' Notes
    1)
        
    Results
    test_median
    Radius = 1, STD threshold = .2: .4% replaced, no significant change to accuracy
    Radius = 1, STD threshold = .25: 2.2% replaced, slight decrease in accuracy
    
    test_mean
    Radius = 1, STD threshold = .25, 2.2% replaced, slight decrease in accuracy
    
    compromise (Ridge/Mean)
    Radius = 1, STD threshold = .25, 2.2% replace, no change
    Radius = 1, STD threshold = .30, 7.32% replaced, slight improvement rmse = .53525
    Radius = 1, STD threshold = .35, 16.583% replaced, improvement      rmse = .534637
    
    compromise (Ridge/Median)
    Radius = 1, STD threshold = .30, 7.32% replaced, slight improvement rmse = .535292
    Radius = 1, STD threshold = .35, 16.583% replaced, improvement      rmse = .534717
    Radius = 1, STD threshold = .35, 28.81% replaced, improvement       rmse = .534212
    
end Notes'''

file_loc = '/home/apark/Homework/ML_Project2/data/'

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
x = import_train['x']
y = import_train['y']
z = import_train['z']

## Standardization
scaler = preprocessing.StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

## PCA and Feature Selection
pca = PCA(n_components=100)
selection = SelectKBest(k=50)
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
flaty = np.zeros([501,1])
combined_features.fit(X_train_scaled,flaty)
print(pca.explained_variance_ratio_) 
X_train_reduced = pca.transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

## Train final Classifier
clf = Ridge(alpha=.5)
clf.fit(X_train_reduced, Y_train_raw)
Y_predicted = clf.predict(X_test_reduced)

## clustering algorithm

RADIUS = 1.0
STD_THRESHOLD = .4
n_train_samples = X_train_raw.shape[0]
n_test_samples = X_test_raw.shape[0]
n_features = provideIdx.shape[0]
n_labels = missIdx.shape[0]

sum = 0
std = np.zeros(X_train_raw.shape)
test_mean = np.zeros([n_test_samples, n_labels])
test_median = np.zeros([n_test_samples, n_labels])
test_std = np.zeros([n_test_samples, n_labels])

for i in range(0,n_labels):
    thisx = x[missIdx[i]]
    thisy = y[missIdx[i]]
    thisz = z[missIdx[i]]
    if( np.mod(i,100) == 0):
        print i, 'out of ', n_labels, 'Percent complete: ', 100.0*i/n_labels
    
    dist = np.zeros([n_features,1])
    for j in range(0,n_features):
        ind = provideIdx[j]
        dist[j] = np.sqrt( (thisx - x[ind])**2 + (thisy - y[ind])**2 + (thisz - z[ind])**2)
    local_index = np.where(dist <= RADIUS)
    k_NN = local_index[0].shape[0]
    sum = sum + k_NN
    
    for n in range(0,n_test_samples):
        if ( k_NN > 2):
            local_voxels = X_test_raw[n,local_index[0]]
            test_mean[n,i] = np.mean(local_voxels)
            test_median[n,i] = np.median(local_voxels)
            test_std[n,i] = np.std(local_voxels)
        else:
            test_std[n,i] = 99
    
print 'average k neighbors is', sum / (i + 1), 'for RADIUS =',RADIUS 

n_replaced = 0
for j in range(0,n_labels): #features     
        for i in range(0,n_test_samples): # samples
            if(test_std[i,j] < STD_THRESHOLD):
                n_replaced = n_replaced + 1
                Y_predicted[i,j] = np.mean([test_median[i,j], Y_predicted[i,j] ])
         
print 'n_replaced',n_replaced, 'out of',(Y_predicted.shape[0]*Y_predicted.shape[1])
print 'or', ((1.0*n_replaced) / (Y_predicted.shape[0]*Y_predicted.shape[1]*1.0)), 'percent'

## Save results to csv
np.savetxt('prediction.csv', Y_predicted, fmt='%.5f',delimiter=',')
