#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter( bumpy_slow,grade_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

'''
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
print(clf.score(features_test,labels_test))
'''

from sklearn.neighbors import KNeighborsClassifier
for i in range(2,50):
	clf=KNeighborsClassifier(n_neighbors=i)
	clf.fit(features_train,labels_train)
	print(i,":",clf.scor e(features_test,labels_test))


'''
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(features_train,labels_train)
print(clf.score(features_test,labels_test))
'''
'''
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(features_train,labels_train)
print(clf.score(features_test,labels_test))
'''



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
