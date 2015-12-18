# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
file_loc = '/home/abhishekb/ML_Project2/data/'
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, chi2
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.cm as cm

# Import test data and labels
import_test = sio.loadmat(file_loc + 'Test.mat')
import_train = sio.loadmat(file_loc + 'Train.mat')
X_train = import_train['Xtrain']
X_testing = import_test['Xtest']
Y_train = import_train['Ytrain']

#Standardization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_testing = scaler.transform(X_testing)

#PCA
pca = PCA(n_components=800)
kpca = KernelPCA(n_components=800, kernel="poly", fit_inverse_transform=True, gamma=None)
#ipca = IncrementalPCA(n_components=800, batch_size=None)

#Selection
selection = SelectKBest(k=450)
selector = SelectPercentile(f_classif, percentile=11)
#combined_features = FeatureUnion([("ipca", ipca), ("univ_select", selection)])
combined_features = FeatureUnion([("pca", pca), ("univ_select", selector)])

combined_features.fit(X_train,Y_train.ravel())
#print(pca.explained_variance_ratio_) 
X_train = combined_features.transform(X_train)
X_testing = combined_features.transform(X_testing)

#k_fold = cross_validation.KFold(len(X_train), 5)
Y_kf = Y_train.ravel()
k_fold = StratifiedKFold(Y_kf, n_folds=10)
#print(k_fold)
#X, X_test, Y, Y_test = cross_validation.train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
#y = Y.ravel()

#X_test = X[401:,:]
#X = X[:400,:]
#X = X[:, :2]
#Y_test = Y[401:,:]
#Y = Y[:400,:]

'''
eventsTrain = import_train['eventsTrain']
subjectsTrain = import_train['subjectsTrain']
x = import_train['x']
y = import_train['y']
z = import_train['z']
'''

for train, test in k_fold:
    X = X_train[train]
    Y = Y_train[train]
    X_test = X_train[test]
    Y_test = Y_train[test]    
    y = Y.ravel()

    C = 1.0  # SVM regularization parameter
    #svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    #rbf_svc = svm.SVC(kernel='rbf', gamma=0.00005, C=50).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=10).fit(X, y)    
    
    for i, clf in enumerate((rbf_svc,lin_svc,poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        z = clf.score(X_test, Y_test)
        if clf == rbf_svc :
            print("RBF",z)
        elif clf == lin_svc:
            print("Linear",z)
        #else :
        #    print("Poly",z)
            
clf = svm.SVC(kernel='rbf', gamma=0.0001, C=10, probability=True).fit(X_train, Y_train.ravel())
#clf = lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train.ravel()) 
Y_test = clf.predict(X_testing)

Y_testing = np.zeros((len(Y_test),3))
for i in range (0,len(Y_test)):
    if (Y_test[i] == 0):
        Y_testing[i,0] = 1
    elif(Y_test[i] == 1):
        Y_testing[i,1] = 1
    elif(Y_test[i] == 3):
        Y_testing[i,2] = 1
     
#print(Y_test)

np.savetxt('final_final/prediction.csv', Y_testing, fmt='%.1f',delimiter=',')