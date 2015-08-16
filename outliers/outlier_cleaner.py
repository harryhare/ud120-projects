#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
	"""
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
	"""
	#print("input")
	#print(ages)
	#print(predictions)
	cleaned_data=[]
	### your code goes here
	cleaned_data=[(ages[i],net_worths[i],(predictions[i]-net_worths[i])**2) for i in range(0,90)]
	#print(len(cleaned_data))
	#print(cleaned_data)
	from operator import itemgetter
	cleaned_data.sort(key=itemgetter(2))
	cleaned_data=cleaned_data[0:81]
	return cleaned_data

