#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

'''my code begin'''
poi_count=0
names=data_dict.keys()
features=data_dict[names[0]].keys()
for name in names:
	if(data_dict[name]['poi']==True):
		poi_count+=1
print(poi_count)

key_person1='LAY KENNETH L'
key_person2='SKILLING JEFFREY K'
key_persons=[key_person1,key_person2]
for name in key_persons:
	person=data_dict[name]
	for feature in features:
		if(person[feature]=="NaN"):
			print(name,feature)

bad_features={}
for name in names:
	person=data_dict[name]
	if(person['poi']==True):
		for feature in features:
			if(person[feature]=="NaN"):
				#print(name,feature)
				if(bad_features.has_key(feature)==False):
					bad_features[feature]=1
				else:
					bad_features[feature]=bad_features[feature]+1
print(bad_features)

#from operator import itemgetter
#temp = sorted(bad_features.items(),key=itemgetter(1))
temp = sorted(bad_features.items(),key=lambda x:x[1])
print(temp)

good_features=[]
for feature in features:
	if(bad_features.has_key(feature)==False):
		good_features.append(feature)
print(good_features)
good_features.remove('poi')
good_features.remove('email_address')
'''only ['total_payments', 'total_stock_value', 'expenses', 'other', 'poi', 'email_address'] '''
'''only three features available : total_payments,total_stock_value, expenses,other'''
features_list=['poi']+good_features

for name in names:
        person=data_dict[name]
        if(person['poi']==True):
                print(name,person['other'])
'''
delete_count=0
for name in names:
        for features in good_features:
                if(data_dict[name][feature]=="NaN"):
                        data_dict.pop(name)
                        delete_count+=1
                        break
print("delete %d person, due to uncomplete information" % delete_count)
'''                





'''my code end'''




### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
'''my code begin'''
from sklearn.cross_validation import train_test_split
features_train, features_test,labels_train , labels_test = train_test_split(
	features,labels,test_size=0.25,random_state=42) 

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
    print("recall",metrics.recall_score(labels_test,pred))
    print("precision",metrics.precision_score(labels_test,pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(labels_test,pred))
    print("done in %0.3fs" % (time()-t0))

param_grid = {
         }
process(GaussianNB,"naive_bayes",param_grid)

param_grid = {
         #'kernel': ['rbf','linear'],
         #'C': [1, 1e2, 1e3, 1e4],
         #'gamma': [0,0.0001, 0.001, 0.01, 0.1],
          }
process(SVC,"svm",param_grid)
param_grid = {
         'criterion':['gini','entropy'],
         'min_samples_split':[2,3,4,5,6]
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

'''my code end'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
