# This file is used to manually determine the epoch that performed best on the validation
# data. There are two approaches: finding the best epoch for each of the X runs, and then
# taking the average of those X epoch values, or selecting the epoch that resulted in the
# best results over all X runs.
#
# This can be useful if you are using combined training on the training and validation sets
# and want to use a set number of epochs to train on, since early stopping wouldn't be very
# effective if the validation data was trained on.

import json
import numpy as np

# This is the function that actually determines the best epoch values. It requires the
# training code to output the file with the number of epochs. The best epoch values
# are then written in a JSON file
#
# Parameters:
#	mode = which method is used; either avg_epoch or best_loss_epoch
#
# Does not return anything

def get_best_epoch(mode):
	fd = open('pretrain_epochs.json', 'r')
	pretrain_data = json.load(fd)
	fd.close()

	fd = open('finetune_epochs.json', 'r')
	finetune_data = json.load(fd)
	fd.close()

	patients = finetune_data.keys()
	runs = len(pretrain_data.keys())

	if mode == 'avg_epoch':
		average_pretrain_epoch = 0
		for i in range(runs):
			average_pretrain_epoch = average_pretrain_epoch + pretrain_data[str(i)]['selected_epoch'] + 1

		average_pretrain_epoch = round(average_pretrain_epoch / runs)

		average_finetune_epochs = {}
		for p in patients:
			average_finetune_epochs[p] = 0
			for i in range(runs):
				average_finetune_epochs[p] = average_finetune_epochs[p] + finetune_data[p][str(i)]['selected_epoch'] + 1
				
			average_finetune_epochs[p] = round(average_finetune_epochs[p] / runs)
			
		pretrain_epoch = average_pretrain_epoch
		finetune_epoch = average_finetune_epochs
		
	elif mode == 'best_loss_epoch':
		max_pretrain_epochs = -1
		for i in range(runs):
			if len(pretrain_data[str(i)]['val_loss']) > max_pretrain_epochs:
				max_pretrain_epochs = len(pretrain_data[str(i)]['val_loss'])
		
		pretrain_avgs = []
		for i in range(max_pretrain_epochs):
			avg = 0		
			for j in range(runs):
				if len(pretrain_data[str(j)]['val_loss']) > i:
					avg = avg + pretrain_data[str(j)]['val_loss'][i][1]
				else:
					avg = avg + 10000
					
			avg = avg / runs
			pretrain_avgs.append(avg)
			
		best_pretrain_epoch = np.argmin(pretrain_avgs).item() + 1
		
		avgs = {}
		
		max_finetune_epochs = {}
		for p in patients:
			max_finetune_epochs[p] = 0
			for i in range(runs):
				if len(finetune_data[p][str(i)]['val_loss']) > max_finetune_epochs[p]:
					max_finetune_epochs[p] = len(finetune_data[p][str(i)]['val_loss'])
					
		finetune_avgs = {}
		for p in patients:
			finetune_avgs[p] = []
			for i in range(max_finetune_epochs[p]):
				avg = 0
				for j in range(runs):
					if len(finetune_data[p][str(j)]['val_loss']) > i:
						avg = avg + finetune_data[p][str(j)]['val_loss'][i][1]
					else:
						avg = avg + 10000
				avg = avg / runs
				finetune_avgs[p].append(avg)
				
		best_finetune_epoch = {}
		for p in patients:
			best_finetune_epoch[p] = np.argmin(finetune_avgs[p]).item() + 1
		
		pretrain_epoch = best_pretrain_epoch
		finetune_epoch = best_finetune_epoch
		
			
	fd = open('epochs.json', 'w')
	json.dump({'pretrain': pretrain_epoch, 'finetune': finetune_epoch}, fd, indent=4)
	fd.close()
	
if __name__ == '__main__':
	mode = 'avg_epoch'
	#mode = 'best_loss_epoch'
	get_best_epoch(mode)
	
	
	
	
	
	
