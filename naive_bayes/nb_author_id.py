#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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



#########################################################
### your code goes here ###
'''
train_size = len(features_train)
word_size = len(features_train[0])
print("train_size:",train_size)
print("word_size:",word_size)
import numpy as np
word_count = {}
word_count[0] = np.zeros(word_size)
word_count[1] = np.zeros(word_size)
for i in range(0,train_size):
	word_count[labels_train[i]]+=features_train[i]

print(word_count[0])
print(word_count[1])
sum=0
for xx in word_count[0]:
	if(xx!=0):
		print(xx)
		sum+=xx
print("sum:",sum)
sum = word_count[0].sum()
word_count[0]=word_count[0]/sum
sum = word_count[1].sum()
word_count[1]=word_count[1]/sum

test_size = len(features_test)


for i in range(0,test_size):
	
'''

from sklearn.naive_bayes import GaussianNB
clf =  GaussianNB()
clf.fit(features_train,labels_train)
result=clf.score(features_test,labels_test)
print(result)
	



#test

#########################################################


