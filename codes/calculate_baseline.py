# This file is used to actually calculate the baseline results

from baselines import avg_carbs_baseline, avg_carbs_by_time_baseline

import json
import numpy as np
import pickle


# This function calculates the baselines and saves them into a JSON file.
#
# Parameters:
#	scenario = prediction scenario; either carbs, bolus, or combo (bolus given carbs)
#	case_name = example case; either unrestricted or inertial
#
# Does not return any values, all results are written in a file.

def calc_baselines(scenario, case_name):
	if scenario == 'carbs':
		patients = ['540', '544', '552', '559', '563', '575', '584', '588', '591', '596']
	elif scenario == 'combo':
		patients = ['540', '544', '552', '559', '563', '575', '584', '588', '591']
	elif scenario == 'bolus':
		patients = ['540', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']
	else:
		return
		
	if case_name == 'unrestricted':
		case = '3'
	elif case_name == 'inertial':
		case = '1'
	else:
		return

	results = {}

	data_file = scenario + '_data.pkl'

	fd = open(data_file, 'rb')
	(data, _) = pickle.load(fd)
	fd.close()

	validation = data['validation']

	(validation_vectors, validation_labels) = validation

	averages = data['averages']
	averages_by_time = data['averages_by_time']

	rmse = 0
	mae = 0
	rmse_time = 0
	mae_time = 0

	for p in patients:
		scaling = data['scaling'][p]
		vx = validation_vectors[p][case]
		vy = validation_labels[p][case]
		labels = (vy * scaling['max']) + scaling['min']
		
		baseline = avg_carbs_baseline(averages[p], labels)
		rmse += baseline[0]
		mae += baseline[1]
		baseline_time = avg_carbs_by_time_baseline(labels, vx, scaling)
		rmse_time += baseline_time[0]
		mae_time += baseline_time[1]
		
	rmse /= len(patients)
	mae /= len(patients)
	rmse_time /= len(patients)
	mae_time /= len(patients)

	results = {'RMSE': rmse, 'MAE': mae, 'RMSE_TIME': rmse_time, 'MAE_TIME': mae_time}

	fd = open(scenario + '_' + case_name + '_' + 'baseline.json', 'w')
	json.dump(results, fd)
	fd.close()
	
if __name__ == '__main__':
	for s in ['carbs', 'bolus', 'combo']:
		for c in ['unrestricted', 'inertial']:
			calc_baselines(s, c)
