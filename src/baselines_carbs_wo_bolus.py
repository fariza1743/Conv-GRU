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
# meal was predicted for every example

def avg_carbs_baseline(avg, labels):
	baseline = np.zeros(shape=labels.shape)
	baseline.fill(avg)

	return baseline
	
# This function calculates the RMSE and MAE as if the average numebr of carbs per
# meal in the time range of the example was predicted for each example

def avg_carbs_by_time_baseline(labels, features, scaling, simple=False):
	baseline = np.zeros(shape=labels.shape[0])
	
	if simple:
		for i in range(len(labels)):
			avg = (features[i][2] * scaling['max']) + scaling['min']
			baseline[i] = avg
	else:
		if 'input3_layer' in features.keys():
			for i in range(len(labels)):
				avg = (features['input3_layer'][i][2] * scaling['max']) + scaling['min']
				baseline[i] = avg
		else:
			for i in range(len(labels)):
				avg = (features['input2_layer'][i][2] * scaling['max']) + scaling['min']
				baseline[i] = avg
		
	return baseline
	
	return baseline
