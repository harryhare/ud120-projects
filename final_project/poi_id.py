#!/usr/bin/python

delete_uncomplete_data = False
show_log = True

def myprint(*argument):
	if(show_log):
		print(argument)

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
data_dict.pop("TOTAL") #get off outlier

poi_count=0
names=data_dict.keys()
features=data_dict[names[0]].keys()
for name in names:
	if(data_dict[name]['poi']==True):
		poi_count+=1
print(poi_count)

new_feature1="fraction_from_poi_to_this_person"
new_feature2="fraction_from_this_person_to_poi"
for name in names:
	person=data_dict[name]
	if(person['to_messages']!="NaN" and person['from_messages']!="NaN"):
		person[new_feature1]=(person["from_poi_to_this_person"]+0.)/person['to_messages']
		person[new_feature2]=(person["from_this_person_to_poi"]+0.)/person['from_messages']
	else:
		person[new_feature1]="NaN"
		person[new_feature2]="NaN"

		

key_person1='LAY KENNETH L'
key_person2='SKILLING JEFFREY K'
key_persons=[key_person1,key_person2]
for name in key_persons:
	person=data_dict[name]
	for feature in features:
		if(person[feature]=="NaN"):
			print(name,feature)

new_feature3="from_kenneth"
new_feature4="from_jeffrey"
new_feature5="to_kenneth"
new_feature6="to_jeffrey"
'''
from_kenneth = open("emails_by_address/from_kenneth.lay@enron.com.txt","r")
from_jeffrey = open("emails_by_address/from_jeff.skilling@enron.com.txt","r")
to_kenneth = open("emails_by_address/to_kenneth.lay@enron.com.txt","r")
to_jeffrey = open("emails_by_address/to_jeff.skilling@enron.com.txt","r")

import re
split_pattern=re.compile("(,\s*)|(\s+)")
def parseOutTo(email):
	email.seek(0)
	all_text=email.read()
	begin=all_text.find("To: ")
	begin+=4
	all_text=all_text[begin:]
	end=all_text.find("Subject:")
	all_text=all_text[:end]
	all_text=re.sub(split_pattern," ",all_text)
	to_list=all_text.split()
	return to_list

def read_email_to_address(email_list):
	count={}
	for path in email_list:
		real_path="../"+path[20:-2]
		print(real_path)
		email = open(real_path,"r")
		temp = parseOutTo(email)
		for address in temp:
			if address in count:
				count[address]+=1
			else:
				count[address]=1
	return count

def add_to_data_dict(feature,dict):
	for name in names:
		person=data_dict[name]
		email=person["email_address"]
		if(email in dict):
			person[feature]=dict[email]
		else:
			person[feature]="NaN"

temp=read_email_to_address(from_kenneth)
add_to_data_dict(new_feature3,temp)
temp=read_email_to_address(from_jeffrey)
add_to_data_dict(new_feature4,temp)
temp=read_email_to_address(to_kenneth)
add_to_data_dict(new_feature5,temp)
temp=read_email_to_address(to_jeffrey)
add_to_data_dict(new_feature6,temp)
'''			

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

if(delete_uncomplete_data):
	delete_count=0
	for name in names:
			for features in good_features:
					if(data_dict[name][feature]=="NaN"):
							data_dict.pop(name)
							delete_count+=1
							break
	print("delete %d person, due to uncomplete information" % delete_count)

#due to so little features left
#now I ignore some pois who hava uncomplete infomation
#find poi with uncomplete feature
pois=[]
for name in names:
	if(data_dict[name]['poi']==True):
		pois.append(name)
print(len(pois))
print(pois)



def find_poi_without_feature(feature):
	names=[]
	for name in pois:
		if(data_dict[name][feature]=='NaN'):
			names.append(name)
	myprint(feature,names);
	return names

feature_list1=["salary","restricted_stock"]
'''
decisiontree recall,precisio 0.2
'''
feature_list2=feature_list1+["bonus"]
'''
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
	Accuracy: 0.83787	Precision: 0.35919	Recall: 0.27550	F1: 0.31183	F2: 0.28897
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
	Accuracy: 0.85113	Precision: 0.41503	Recall: 0.28450	F1: 0.33759	F2: 0.30360
	Total predictions: 15000	True positives:  569	False positives:  802	False negatives: 1431	True negatives: 12198
'''
feature_list3=feature_list2+["to_messages","from_messages",
 "from_poi_to_this_person","from_this_person_to_poi","shared_receipt_with_poi",
  new_feature1,new_feature2]
'''
After scaler:
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=6, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
	Accuracy: 0.93057	Precision: 0.51163	Recall: 0.61600	F1: 0.55898	F2: 0.59185
[('fraction_from_poi_to_this_person', 0.45804342800706016), 
('total_stock_value', 0.23487292711823565),
 ('expenses', 0.12970722666215376),
 ('from_this_person_to_poi', 0.12954555710659485),
 ('total_payments', 0.047830861105955622),
 ('shared_receipt_with_poi', 0.0), 
 ('from_poi_to_this_person', 0.0),
 ('from_messages', 0.0), 
 ('to_messages', 0.0),
 ('bonus', 0.0),
 ('restricted_stock', 0.0),
 ('salary', 0.0),
 ('other', 0.0),
 ('poi', 0.0)]
'''
feature_list4=feature_list3+[new_feature3,new_feature4,new_feature5,new_feature6]

feature_test=feature_list3 # can be tuned 

name_set=set()
for feature in  feature_test:
	temp=find_poi_without_feature(feature)
	name_set=name_set.union(set(temp))
features_list+=feature_test
for name in name_set:
	data_dict.pop(name)
	myprint("***delete:"+name)



'''my code end'''




### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
'''do min_max_scalar'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#scaler.fit_transform(features)
'''done'''


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
         'kernel': ['rbf'],
         'C': [1, 1e2],
         'gamma': [0, 0.001, 0.01, 0.1],
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

clf=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=6, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
'''my code end'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
