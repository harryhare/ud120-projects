#!/usr/bin/python 

""" 
    skeleton code for k-means clustering mini-project

"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
#salary
#deferral_payments
#total_payments
#exercised_stock_options
'''
'salary', 'deferral_payments', 'total_payments',
'exercised_stock_options',
'bonus',
'restricted_stock',
'restricted_stock_deferred',
'total_stock_value',
'expenses', 
 'director_fees',
 'deferred_income',
 'long_term_incentive',
]

'''
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3="total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2,feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )
print(poi)

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, line below assumes 2 features)
for f1, f2 ,f3 in finance_features:
    plt.scatter( f1, f2 ,f3)
plt.show()



from sklearn.cluster import KMeans
features_list = ["poi", feature_1, feature_2]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data2 )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )
Draw(pred, finance_features, poi, mark_poi=True,name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
'''
features_list = ["poi", feature_1, feature_2,feature_3]
data2 = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data2 )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )
Draw(pred, finance_features, poi, mark_poi=True,name="clusters_before_scaling.pdf", f1_name=feature_1, f2_name=feature_2)
'''
def find_min_max(key,data_dict):
    names=data_dict.keys()
    for i in range(0,len(names)):
        temp=data_dict[names[i]][key]
        if(temp!='NaN'):
            salary_min=data_dict[names[i]][key]
            salary_max=salary_min
            break
    for i in range(0,len(names)):
        temp=data_dict[names[i]][key]
        if(temp=='NaN'):
            continue
        if(temp>salary_max):
            salary_max=temp
        if(temp<salary_min):
            salary_min=temp
    return (salary_min,salary_max)
salary_min,salary_max=find_min_max("salary",data_dict)
eso_min,eso_max=find_min_max("exercised_stock_options",data_dict)

def my_scaler(value,min,max):
    return(value-min+0.0)/(max-min)
print(200000,my_scaler(200000.0,salary_min,salary_max))
print((200000.0-salary_min+0.0)/(salary_max-salary_min))
print(1000000,my_scaler(1000000.0,eso_min,eso_max))
print((1000000.0-eso_min+0.0)/(eso_max-eso_min))
print("salary_min",salary_min)
print("salary_max",salary_max)
print("eso_min",eso_min)
print("eso_max",eso_max)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(finance_features)
import numpy as np
x=np.array([[200000.,1000000.]])
y=scaler.transform(x)
print(y)






try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"





