#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



#clf = classify(features_train, labels_train)
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import grid_search
def process(method,name,param_grid,**argument):
    t0=time()
    clf=grid_search.GridSearchCV(method(**argument),param_grid)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    print(clf.best_estimator_)
    print("accuracy",metrics.accuracy_score(labels_test,pred))
    print("done in %0.3fs" % (time()-t0))
    prettyPicture(clf, features_test, labels_test,name)

param_grid = {
         }
process(GaussianNB,"naive_bayes",param_grid)
param_grid = {
         'kernel': ['rbf','linear'],
         'C': [1, 1e2, 1e3, 1e4],
          'gamma': [0,0.0001, 0.001, 0.01, 0.1],
          }
process(SVC,"svm",param_grid)
param_grid = {
         'criterion':['gini','entropy'],
         'min_samples_split':[2,4,8,16,32,64]
          }
process(DecisionTreeClassifier,"DecisionTree",param_grid)

param_grid = {
         'n_neighbors' : [1,3,5,7,9,11],
         }
process(KNeighborsClassifier,"KNN",param_grid)

param_grid = {
         'n_estimators' : [2,4,6,8,10],
    }
process(RandomForestClassifier,"RandomForest",param_grid)

param_grid = {
         'n_estimators' : [10,20,30,40,50],
    }
process(AdaBoostClassifier,"AdaBoost",param_grid)

### draw the decision boundary with the text points overlaid
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())




