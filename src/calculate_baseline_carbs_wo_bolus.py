from baselines_carbs_wo_bolus import avg_carbs_baseline, avg_carbs_by_time_baseline

import json
import numpy as np
import pickle

def RMSE(predictions, labels):
	rmse = (predictions - labels) ** 2
	rmse = np.mean(rmse)
	rmse = np.sqrt(rmse)
	return rmse
	
	
def MAE(predictions, labels):
	mae = np.abs(predictions - labels)
	mae = np.mean(mae)
	return mae

def calc_baselines(scenario, case_name):
	if scenario == 'carbs':
		patients = ['540', '544', '552', '559', '563', '575', '584', '588', '591', '596']
	elif scenario == 'combo':
		patients = ['540', '544', '552', '559', '563', '575', '584', '588', '591']
	else:
		patients = ['540', '552', '559', '563', '570', '575', '584', '588', '591', '596']
		
	if '540' in patients:
		patients.remove('540')
	if '596' in patients:
		patients.remove('596')
		
	if case_name == 'unrestricted':
		case = '3'
	else:
		case = '1'

	results = {}

	data_file = scenario + '_data.pkl'

	fd = open(data_file, 'rb')
	(data, _) = pickle.load(fd)
	fd.close()

	validation = data['testing']

	(validation_vectors, validation_labels) = validation

	averages = data['averages']
	averages_by_time = data['averages_by_time']

	rmse = 0
	mae = 0
	rmse_time = 0
	mae_time = 0

	predictions = None
	predictions_time = None
	all_labels = None
	for p in patients:
		scaling = data['scaling'][p]
		vx = validation_vectors[p][case]
		vy = validation_labels[p][case]
		labels = (vy * scaling['max']) + scaling['min']
		
		baseline = avg_carbs_baseline(averages[p], labels)
		baseline_time = avg_carbs_by_time_baseline(labels, vx, scaling)
		if predictions is None:
			predictions = baseline
			predictions_time = baseline_time
			all_labels = labels
		else:
			predictions = np.append(predictions, baseline)
			predictions_time = np.append(predictions_time, baseline_time)
			all_labels = np.append(all_labels, labels)
	
	rmse = RMSE(predictions, all_labels)
	mae = MAE(predictions, all_labels)
	rmse_time = RMSE(predictions_time, all_labels)
	mae_time = MAE(predictions_time, all_labels)

	results = {'RMSE': rmse, 'MAE': mae, 'RMSE_TIME': rmse_time, 'MAE_TIME': mae_time}

	fd = open(scenario + '_' + case_name + '_' + 'baseline.json', 'w')
	json.dump(results, fd)
	fd.close()
	
for c in ['unrestricted', 'inertial']:
	calc_baselines('carbs', c)
