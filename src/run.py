# This file will pretrain, finetune, and evaluate models for the desired scenario
# and example class. This is the main function used for experiments as all steps
# are done automatically.

from pretrain import pretrain
from finetune import finetune
from testing import evaluate, compute_final_results
from load_data import load_data
import json
import argparse

# This is the main function to load the data, pretrain, finetune, and evaluate models.
#
# Parameters:
#	data_file = the name of the pickle file with the data
#
#	hyper_parameters = dictonary of hyper-parameters
#
#	patient_list = the list of patients to finetune and evaluate models for.
#
#	unrestricted = True if unrestricted, False for inertial
#
#	combined = True if training on both the training and validation data
#
#	runs = The number of generic models to pretrain and the number of models to be finetuned
# 	and evaluated for each subject
#
#	specific_epochs = A dictonary with the number of epochs to train for each subject.
#	None if you do not want to specify number of epochs. This would likely only be
#   needed if you are not using early stopping and want to set per-patient maximum epochs.
#
#	specific_epochs_pretrain = Same as specific epochs except for the pretraining step.
#
#	track_epochs = True if you want the function to output the number of epochs that were
# 	used in training. False otherwise. You would likely only want this information if
#	early stopping is used.
#
#	tensorboard = True if you want to save the tensorboard logs, false otherwise
#
#	combo = True if bolus given carbs scenario. False otherwise.

def run(data_file, 
		hyper_parameters, 
		patient_list,
		unrestricted=True, 
		combined=False,
		runs=1,
		specific_epochs=None,
		specific_epochs_pretrain=None,
		track_epochs=False,
		tensorboard=True,
		combo=False):
	
	pretrain_data, _ = load_data(data_file, 
							  pretrain=True,
							  unrestricted=unrestricted, 
							  combined=combined)
							  
	pretrain(pretrain_data, 
			 hyper_parameters, 
			 unrestricted=unrestricted, 
			 runs=runs,
			 specific_epochs=specific_epochs_pretrain,
			 track_epochs=track_epochs,
			 tensorboard=tensorboard,
			 combo=combo)
			 
	finetune_data = {}
	scaling = {}
	for p in patient_list:
		(finetune_data[p], scaling[p]) = load_data(data_file,
												   pretrain=False,
												   patient=p,
												   unrestricted=unrestricted,
												   combined=combined)
												   
	finetune(finetune_data, 
			 hyper_parameters, 
			 patient_list,
			 unrestricted=unrestricted,
			 runs=runs,
			 specific_epochs=specific_epochs,
			 track_epochs=track_epochs,
			 tensorboard=tensorboard)
			 
	(results, best_models) = evaluate(finetune_data,
									  patient_list,
									  scaling,
									  batch_size=hyper_parameters['batch_size'],
									  testing_set='validation',
									  unrestricted=unrestricted,
									  runs=runs,
									  tensorboard=tensorboard)
									  
	final_results = compute_final_results(results, best_models, patient_list)
	
	return final_results
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--scenario', default='carbs', type=str)
	parser.add_argument('--case', default='unrestricted', type=str)
	parser.add_argument('--time', default=0, type=int)
	parser.add_argument('--combined', default='False', type=str)
	parser.add_argument('--runs', default=5, type=int)
	args = parser.parse_args()
	
	scenario = args.scenario
	
	time = args.time
	if time == 0:
		time = None
		
	if args.combined == 'True':
		combined = True
	else:
		combined = False
		
	case = args.case
	if case == 'unrestricted':
		unrestricted = True
	else:
		unrestricted = False
		
	runs = args.runs
	
	if scenario == 'carbs':
		patient_list = ['540', '544', '552', '559', '563', '575', '584', '588', '591', '596']
	elif scenario == 'combo':
		patient_list = ['540', '544', '552', '559', '563', '575', '584', '588', '591']
	else:
		patient_list = ['540', '552', '559', '563', '570', '575', '584', '588', '591', '596']
		
	if scenario == 'combo':
		combo = True
	else:
		combo = False
		
	if time is None:
		data_file = scenario + '_data.pkl'
	else:
		data_file = scenario + '_data' + str(time) + '.pkl'
		
	if combined:
		track_epochs = False
		specify_epochs = True
	else:
		track_epochs = True
		specify_epochs = False
	
	if specify_epochs:
		fd = open('epochs.json', 'r')
		epochs = json.load(fd)
		fd.close()
		specific_epochs_pretrain = epochs['pretrain']
		specific_epochs = epochs['finetune']
	else:
		specific_epochs_pretrain = None
		specific_epochs = None
	
	fd = open(scenario + '_hyper_parameters_' + case + '.json', 'r')
	hyper_parameters = json.load(fd)
	fd.close()
	
	results = run(data_file, 
				  hyper_parameters, 
				  patient_list, 
				  runs=runs,
				  unrestricted=unrestricted,
				  combined=combined,
				  track_epochs=track_epochs,
				  specific_epochs=specific_epochs,
				  specific_epochs_pretrain=specific_epochs_pretrain,
				  combo=combo
				  )

	fd = open('run_results.json', 'w')
	json.dump(results, fd)
	fd.close()
