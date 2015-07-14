#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train=features_train[0:len(features_train)/100]
#labels_train=labels_train[0:len(labels_train)/100]

from sklearn.svm  import SVC
def data_process(C):
	clf=SVC(C,kernel='rbf')
	t0=time()
	clf.fit(features_train,labels_train)
	print("trainint time:",round(time()-t0,3),'s')
	t0=time()
	accuracy=clf.score(features_test,labels_test) 
	print("testint time:",round(time()-t0,3),'s')
	print("C=",C,",accuracy=",accuracy)


#########################################################
### your code goes here ###
'''
data_process(1.0)
data_process(10.0)
data_process(100.0)
data_process(1000.0)
data_process(10000.0)
data_process(100000.0)
data_process(1000000.0)
'''


#data_process(10000.0)
clf=SVC(10000,kernel='rbf')
t0=time()
clf.fit(features_train,labels_train)
print("trainint time:",round(time()-t0,3),'s')
t0=time()
prediction=clf.predict(features_test) 
print("testint time:",round(time()-t0,3),'s')
print(prediction.sum())
	
#########################################################


