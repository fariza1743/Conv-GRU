# This file contains the functions required to calculate the global and time-of-day
# baselines (avg_carbs_baseline and avg_carbs_by_time_baseline, respectively)

import numpy as np

def RMSE(predictions, labels):
	rmse = (predictions - labels) ** 2
	rmse = np.mean(rmse)
	rmse = np.sqrt(rmse)
	return rmse
	
	
def MAE(predictions, labels):
	mae = np.abs(predictions - labels)
	mae = np.mean(mae)
	return mae

# This function calculates the RMSE and MAE as if the average number of carbs per
# meal was predicted for every example. Used for the global baseline.
#
# Parameters: 
# 	avg = the value to be used as the baseline prediction for all examples
#	labels = the actual labels
#
# Returns the RMSE and MAE

def avg_carbs_baseline(avg, labels):
	baseline = np.zeros(shape=labels.shape)
	baseline.fill(avg)
		
	rmse = RMSE(baseline, labels)
	mae = MAE(baseline, labels)

	return rmse, mae
	
# This function calculates the RMSE and MAE as if the average numebr of carbs per
# meal in the time range of the example was predicted for each example. Used for
# the time-of-day baseline. The time-of-day baseline is included in the features
# parameter and is scaled up and used as the baseline prediction for the example
#
# Parameters:
# 	labels = the actual labels of the examples
#	features = the same input features that would be given to the model
#	scaling = the parameters used to scale the data
#
# Returns the RMSE and MAE

def avg_carbs_by_time_baseline(labels, features, scaling):
	baseline = np.zeros(shape=labels.shape[0])
	
	if 'input3_layer' in features.keys():
		for i in range(len(labels)):
			avg = (features['input3_layer'][i][2] * scaling['max']) + scaling['min']
			baseline[i] = avg
	else:
		for i in range(len(labels)):
			avg = (features['input2_layer'][i][2] * scaling['max']) + scaling['min']
			baseline[i] = avg
		
	rmse = RMSE(baseline, labels)
	mae = MAE(baseline, labels)
	
	return rmse, mae
