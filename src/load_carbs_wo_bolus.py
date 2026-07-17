# This is the file that loads the data from the raw XML files into examples in a
# pickle file.

import datetime as dt
import numpy as np
import csv
import pickle
import xml.etree.cElementTree as XMLParser
import json

# Some global variables that are useful for several functions in this file
index = {'bgl': 0, 'basal': 1, 'bolus': 2, 'bolus_carbs': 3, 'bolus_type': 4, 'meal': 5}
bolus_type_values = {'normal': 0, 'square': 1, 'normal dual': 2, 'square dual': 3}

time_frames = [((0, 6), '00-06'), 
			   ((6, 10), '06-10'),
			   ((10, 14), '10-14'), 
			   ((14, 18), '14-18'),
			   ((18, 24), '18-00')
			  ]

# This function parses and loads the contents of the XML file into memory. Originally
# written by Sadegh Mirshekarian.
#
# Parameters:
#	fileNamesAndIDs = list of pairs in the format (subject ID, filename0)
#
#	res = time resolution in minutes. Should likely remain at 5.
#
#	includeCats = list of included categories. BGL is automatically included and
#	does not need to be in this list.
#
#	dtFormat = format string for the timestamp in the XML files
#
#	verbose = True if you want to see more specific messages during loading.
#
# Returns:
#	a dictionary of data, the timestamps of each timestep, and a mask indicated if
#	the BGL was interpolated at that timestep or if it is real.	
def load_from_xml(fileNamesAndIDs, res=5, includeCats=[],
                  dtFormat='%d-%m-%Y %H:%M:%S', verbose=True):            
  data = {}
  timeStamps = {}
  isRealBGL = {}
  
  def _valueAtTimeInCat(tm, category, threshold):
    """
    Helper function that searches in a 'category' (which is just a dictionary of
    datetime:value pairs) and checks if 'tm' is in the keys within a 'threshold'.
    """
    delta1 = dt.timedelta(minutes=1)
    for r in range(threshold+1):
        if (tm + r*delta1) in category:
            return category[tm + r*delta1]
        elif (tm - r*delta1) in category:
            return category[tm - r*delta1]
    return 0

  if not isinstance(includeCats, list):
	  raise Exception('The category list to include must be a LIST of category labels.')


  for pID, fname in fileNamesAndIDs:
	  print("\n> Parsing file {} for patient '{}'...".format(fname, pID))
	  xmlRoot = XMLParser.parse(fname).getroot()
	  xmlIter = xmlRoot.iter()

	  item = next(xmlIter)
	  while item.tag != 'glucose_level':
		  item = next(xmlIter)

	  print(" > Parsing category 'bgl'...")
	  categories      = [] # list of encountered categories
	  bgl             = []
	  isRealBGL[pID]  = []
	  categories.append(bgl)
	  timeStamps[pID] = []

	  item = next(xmlIter)
	  prevLevel = 0
	  prevDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)

	  while item.tag == 'event':
		  curDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
		  curLvl = float(item.attrib['value'])

		  diff = (curDT - prevDT).total_seconds() / 60

		  if diff > res:
			  if verbose:
				  print(" Warning: bgl data discontinuity at {}".format(prevDT))
			  s = int(diff // res - (diff % res == 0))
			  m = (curLvl - prevLevel) / diff
			  for i in range(s):
				  prevLevel += m * res
				  bgl.append(prevLevel)
				  isRealBGL[pID].append(False)
				  timeStamps[pID].append(prevDT + dt.timedelta(minutes=res*(i+1)))

		  bgl.append(curLvl)
		  isRealBGL[pID].append(True)
		  timeStamps[pID].append(curDT)

		  prevDT = curDT
		  prevLevel = curLvl
		  item = next(xmlIter)

	  if 'timeOfDay_real' in includeCats:
		  print(" > Time of day is being included as total seconds since midnight...")
		  timeOfDay_real = {ts:(ts-ts.replace(minute=0, hour=0)).total_seconds() for ts in timeStamps[pID]}
		  categories.append(timeOfDay_real)

	  if 'timeOfDay_ticks' in includeCats:
		  print(" > Time of day is being included as one feature with ticks on the hour...")
		  timeOfDay_ticks = {}
		  prevHour = -1
		  for ts in timeStamps[pID]:
			  thisHour = ts.hour
			  timeOfDay_ticks[ts] = (thisHour != prevHour)*1
			  prevHour = ts.hour

		  categories.append(timeOfDay_ticks)

	  while item.tag:
		  try:
			  if item.tag=='finger_stick' and 'finger_stick' in includeCats:
				  print(" > Parsing category 'finger_stick'...")
				  fstick = {}
				  categories.append(fstick)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  fstick[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  elif item.tag=='basal' and 'basal' in includeCats:
				  print(" > Parsing category 'basal'...")
				  basal = {}
				  categories.append(basal)
				  beginDT = None
				  item = next(xmlIter)
				  _value = None
				  while item.tag=='event':
					  endDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  if beginDT:
						  while beginDT <= endDT:
							  basal[beginDT] = _value
							  beginDT = beginDT + dt.timedelta(minutes=1)
					  beginDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  _value = item.attrib['value']
					  item = next(xmlIter)

			  elif item.tag=='temp_basal' and 'basal' in includeCats:
				  print(" > Parsing category 'temp_basal' and correcting 'basal'...")
				  item = next(xmlIter)
				  while item.tag=='event':
					  beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
					  endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
					  while beginDT <= endDT:
						  basal[beginDT] = item.attrib['value']
						  beginDT = beginDT + dt.timedelta(minutes=1)
					  item = next(xmlIter)

			  elif item.tag=='bolus' and 'bolus' in includeCats:
				  print(" > Parsing category 'bolus'...")
				  bolus = {}
				  bolus_carbs = {}
				  bolus_type = {}
				  categories.append(bolus)
				  categories.append(bolus_carbs)
				  categories.append(bolus_type)
				  item = next(xmlIter)
				  
				  while item.tag=='event':
					  beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
					  endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
					  num_minutes = max(1, (endDT-beginDT).total_seconds() / 60)
					  bolus_per_minute = float(item.attrib['dose']) / num_minutes
					  while beginDT <= endDT:
						  bolus[beginDT] = min(res, num_minutes) * bolus_per_minute
						  bolus_type[beginDT] = bolus_type_values[item.attrib['type']]
						  bolus_carbs[beginDT] = float(item.attrib['bwz_carb_input'])
						 
						  beginDT = beginDT + dt.timedelta(minutes=res)
						  num_minutes = (endDT-beginDT).total_seconds() / 60
					  item = next(xmlIter)

			  elif item.tag=='meal' and 'meal' in includeCats:
			      print(" > Parsing category 'meal'...")
			      meal = {}
			      categories.append(meal)
			      item = next(xmlIter)
			      while item.tag=='event' or item.tag=='food':
				      if item.tag=='event':
					      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					      carb = float(item.attrib['carbs'])
					      meal[DT] = carb
				      item = next(xmlIter)

			  elif item.tag=='sleep' and 'sleep' in includeCats:
				  print(" > Parsing category 'sleep'...")
				  sleep = {}
				  categories.append(sleep)
				  item = next(xmlIter)
				  while item.tag=='event':
					  beginDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
					  endDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
					  if beginDT > endDT:
						  beginDT, endDT = endDT, beginDT
					  if endDT-beginDT > dt.timedelta(hours=12) and verbose:
						  print(" Warning: long sleep detected from {} to {}, duration={} hr"
								  .format(beginDT, endDT, (endDT-beginDT).total_seconds()/3600))
					  while beginDT <= endDT:
						  sleep[beginDT] = 1
						  beginDT = beginDT + dt.timedelta(minutes=1)
					  item = next(xmlIter)

			  elif item.tag=='work' and 'work' in includeCats:
				  print(" > Parsing category 'work'...")
				  work = {}
				  categories.append(work)
				  item = next(xmlIter)
				  while item.tag=='event':
					  beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
					  endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
					  if endDT-beginDT > dt.timedelta(hours=12) and verbose:
						  print(" Warning: long work detected from {} to {}, duration={} hr"
								  .format(beginDT, endDT, (endDT-beginDT).total_seconds()/3600))
					  while beginDT <= endDT:
						  work[beginDT] = float(item.attrib['intensity'])
						  beginDT = beginDT + dt.timedelta(minutes=1)
					  item = next(xmlIter)

			  elif item.tag=='infusion_set' and 'infusion_set' in includeCats:
				  print(" > Parsing category 'infusion_set'...")
				  infusion_set = {}
				  categories.append(infusion_set)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  infusion_set[DT] = 1
					  item = next(xmlIter)

			  elif item.tag=='hypo_event' and 'hypo_event' in includeCats:
				  print(" > Parsing category 'hypo_event'...")
				  hypo_event = {}
				  categories.append(hypo_event)
				  item = next(xmlIter)
				  while item.tag=='event' or item.tag=='symptom':
					  if item.tag=='event':
						  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
						  hypo_event[DT] = 1
					  item = next(xmlIter)

			  elif item.tag=='hypo_action' and 'meal' in includeCats:
				  print(" > Parsing category hypo_action and adding to 'meal...")
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  if DT in meal and verbose:
						  print(" Warning: hypo action already in 'meals' at {} with carbs={}, replacing with {}..."
								  .format(DT, meal[DT], item.attrib['carbs']))
					  meal[DT] = item.attrib['carbs']
					  item = next(xmlIter)

			  elif item.tag=='exercise' and 'exercise' in includeCats:
				  print(" > Parsing category 'exercise'...")
				  exercise = {}
				  categories.append(exercise)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  duration = float(item.attrib['duration'])
					  intensity = float(item.attrib['intensity'])
					  for i in range(int(duration)+1):
						  exercise[DT+dt.timedelta(minutes=i)] = intensity
					  item = next(xmlIter)

			  elif item.tag=='basis_heart_rate' and 'basis_heart_rate' in includeCats:
				  print(" > Parsing category 'basis_heart_rate'...")
				  hrate = {}
				  categories.append(hrate)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  hrate[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  elif item.tag=='basis_gsr' and 'basis_gsr' in includeCats:
				  print(" > Parsing category 'basis_gsr'...")
				  gsr = {}
				  categories.append(gsr)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  gsr[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  elif item.tag=='basis_skin_temperature' and 'basis_skin_temperature' in includeCats:
				  print(" > Parsing category 'basis_skin_temperature'...")
				  skin_temp = {}
				  categories.append(skin_temp)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  skin_temp[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  elif item.tag=='basis_air_temperature' and 'basis_air_temperature' in includeCats:
				  print(" > Parsing category 'basis_air_temperature'...")
				  air_temp = {}
				  categories.append(air_temp)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  air_temp[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  elif item.tag=='basis_steps' and 'basis_steps' in includeCats:
				  print(" > Parsing category 'basis_steps'...")
				  steps = {}
				  categories.append(steps)
				  item = next(xmlIter)
				  while item.tag=='event':
					  DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
					  steps[DT] = float(item.attrib['value'])
					  item = next(xmlIter)

			  else:
				  item = next(xmlIter)

		  except StopIteration:
			  break

	  print(" > Parsing done, now merging the data into a single matrix...")

	  N = len(bgl)
	  D = len(categories)
	  data[pID] = np.zeros((N, D))

	  for t, tm in enumerate(timeStamps[pID]):
		  data[pID][t, 0] = bgl[t]
		  for c, cat in enumerate(categories[1:]):
			  data[pID][t, c+1] = _valueAtTimeInCat(tm, cat, threshold=2)

  return data, timeStamps, isRealBGL
  
# This function calculates some basic statistics about the data, including values
# used to scale the data.
#
# Parameters:
#	patients = list of patient ids to calculate the stats for.
#
#	files = dictonary of file names of the form {'training': ['559-training.xml', ...]}.
#	This dictonary should include file names for training, testing, and validation.
#	The order of the lists should follow the order in the patients parameter.
#	For example, if the first patient ID in patients is '559', the first filename
# 	in files['training'] should be the file name containing subject 559's training data.
#
#	event_type = 'meal' or 'bolus'. The function will then calculate stats about
#	either the meals or boluses in the data depending on the selection here.
#
#	time_res = time resolution in minutes. Should probably stay at 5
#
#	meal_bolus_distance = The number of minutes between a meal and it's associated
#	bolus after pre-processing.
#
#	use_bolus_carbs = If true, if a meal moved in the pre-processing step, also
#	replace the number of carbs in the meal to the value that was entered into the
#	bolus wizard.
#
#	combined = True if you plan to use the combination of training and validation data
#	to train the models. False otherwise
#
#	save = True if you would like to save these stats into a pickle file.
#
# Returns:
#	The calculated stats
	
def calculate_stats(patients, 
                    files, 
                    event_type='meal', 
                    time_res=5, 
                    meal_bolus_distance=10,
                    use_bolus_carbs=False,
                    combined=True, 
                    save=False
                   ):

	bgl_scaling = {}
	meal_scaling = {}
	insulin_scaling = {}
	pretrain_bgl_scaling = {'min': float('inf'), 'max': float('-inf')}
	pretrain_meal_scaling = {'min': float('inf'), 'max': float('-inf')}
	pretrain_insulin_scaling = {'min': float('inf'), 'max': float('-inf')}
	
	maxs = {}
	mins = {}
	
	pretrain_maxs = [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')]
	pretrain_mins = [float('inf'), float('inf'), float('inf'), float('inf'),  float('inf'), float('inf')]

	avgs = {}
	avgs_by_time = {}
					
	stats = {}
	
	stats['by_patient'] = {}
	for p in patients:
		stats['by_patient'][p] = {}
		for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
			stats['by_patient'][p][c] = {}
			for s in ['amount', 'average', 'max', 'min', 'med']:
				stats['by_patient'][p][c][s] = 0
	
	stats['total'] = {}
	for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
		stats['total'][c] = {}
		for s in ['amount', 'average', 'max', 'min', 'med']:
			stats['total'][c][s] = 0
				
	all_data = {}
	
	all_data['total'] = {'bgl': None, 'basal': None, 'bolus': None, 'meal': None}
	for p in patients:
		all_data[p] = {'bgl': None, 'basal': None, 'bolus': None, 'meal': None}
	
	for d in ['training', 'testing', 'validation']:
		dataRaw, timeStamps, isRealBGL = load_from_xml(zip(patients, files[d]), time_res,
				                                       ['bgl', 'meal', 'bolus', 'basal'], verbose=False)
				                                       
		dataRaw = pre_process_data(patients, 
		                           dataRaw, 
		                           meal_bolus_distance=meal_bolus_distance, 
		                           time_res=time_res, 
		                           use_bolus_carbs=use_bolus_carbs
				                  )             
				                                       
		stats[d] = {}
       
		for p in patients:
			stats[d][p] = {}
			for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
				stats[d][p][c] = {'amount': 0, 'average': 0, 'max': 0, 'min': 0, 'med': 0, 'std': 0}
			
		nonzero_data = {}	
		for p in patients:
			for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
				nonzero_data = dataRaw[p][:, index[c]]
				nonzero_data = nonzero_data[nonzero_data != 0]
				if nonzero_data.size == 0:
					stats[d][p][c]['amount'] = 0
					stats[d][p][c]['average'] = 0
					stats[d][p][c]['max'] = 0
					stats[d][p][c]['min'] = 0
					stats[d][p][c]['med'] = 0
					stats[d][p][c]['std'] = 0
				else:
					stats[d][p][c]['amount'] = nonzero_data.shape[0]
					stats[d][p][c]['average'] = np.average(nonzero_data)
					stats[d][p][c]['max'] = np.max(nonzero_data)
					stats[d][p][c]['min'] = np.min(nonzero_data)
					stats[d][p][c]['med'] = np.median(nonzero_data)
					stats[d][p][c]['std'] = np.std(nonzero_data)
				if d == 'training':
					all_data[p][c] = nonzero_data
				else:
					all_data[p][c] = np.append(all_data[p][c], nonzero_data)
				
		for p in patients:
			for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
				if all_data[p][c].size == 0:
					stats['by_patient'][p][c]['amount'] = 0
					stats['by_patient'][p][c]['average'] = 0
					stats['by_patient'][p][c]['max'] = 0
					stats['by_patient'][p][c]['min'] = 0
					stats['by_patient'][p][c]['med'] = 0
					stats['by_patient'][p][c]['std'] = 0
				else:
					stats['by_patient'][p][c]['amount'] = all_data[p][c].shape[0]
					stats['by_patient'][p][c]['average'] = np.average(all_data[p][c])
					stats['by_patient'][p][c]['max'] = np.max(all_data[p][c])
					stats['by_patient'][p][c]['min'] = np.min(all_data[p][c])
					stats['by_patient'][p][c]['med'] = np.median(all_data[p][c])
					stats['by_patient'][p][c]['std'] = np.std(all_data[p][c])
					
		stats[d]['total'] = {}
		for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
			stats[d]['total'][c] = {'amount': 0, 'average': 0, 'max': 0, 'min': 0, 'med': 0, 'std': 0}
		
		combined_data = {}
		for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
			first = True
			for p in patients:
				nonzero_data = dataRaw[p][:, index[c]]
				nonzero_data = nonzero_data[nonzero_data != 0]
				if first:
					combined_data[c] = nonzero_data
					first = False
				else:
					combined_data[c] = np.append(combined_data[c], nonzero_data)
					
			stats[d]['total'][c]['amount'] = combined_data[c].shape[0]
			stats[d]['total'][c]['average'] = np.average(combined_data[c])
			stats[d]['total'][c]['max'] = np.max(combined_data[c])
			stats[d]['total'][c]['min'] = np.min(combined_data[c])
			stats[d]['total'][c]['med'] = np.median(combined_data[c])
			stats[d]['total'][c]['std'] = np.std(combined_data[c])
			if d == 'training':
				all_data['total'][c] = combined_data[c]
			else:
				all_data['total'][c] = np.append(all_data['total'][c], combined_data[c])
				
		for c in ['bgl', 'basal', 'bolus', 'bolus_carbs', 'meal']:
			stats['total'][c]['amount'] = all_data['total'][c].shape[0]
			stats['total'][c]['average'] = np.average(all_data['total'][c])
			stats['total'][c]['max'] = np.max(all_data['total'][c])
			stats['total'][c]['min'] = np.min(all_data['total'][c])
			stats['total'][c]['med'] = np.median(all_data['total'][c])
			stats['total'][c]['std'] = np.std(all_data['total'][c])
			
			
		if d == 'training':
			for p in patients:
				mins[p] = dataRaw[p].min(axis=0)
				maxs[p] = dataRaw[p].max(axis=0)
				basal_min = mins[p][index['basal']]
				basal_max = maxs[p][index['basal']]
				bolus_min =  mins[p][index['bolus']]
				bolus_max = maxs[p][index['bolus']]
				carb_min = mins[p][index['meal']]
				carb_max = maxs[p][index['meal']]
				bolus_carb_min = mins[p][index['bolus_carbs']]
				bolus_carb_max = maxs[p][index['bolus_carbs']]
				
				mins[p][index['basal']] = basal_min + bolus_min
				mins[p][index['bolus']] = basal_min + bolus_min
				maxs[p][index['basal']] = basal_max + bolus_max
				maxs[p][index['bolus']] = basal_max + bolus_max
				maxs[p][index['meal']] = max(carb_max, bolus_carb_max)
				maxs[p][index['bolus_carbs']] = max(carb_max, bolus_carb_max)
				mins[p][index['meal']] = min(carb_min, bolus_carb_min)
				mins[p][index['bolus_carbs']] = min(carb_min, bolus_carb_min)
				
				
				maxs[p][index['bolus_type']] = 1
				mins[p][index['bolus_type']] = 0
				
				meal_scaling[p] = {'min': mins[p][index['meal']], 'max': maxs[p][index['meal']]}
				bgl_scaling[p] = {'min': mins[p][index['bgl']], 'max': maxs[p][index['bgl']]}
				insulin_scaling[p] = {'min': mins[p][index['bolus']], 'max': maxs[p][index['bolus']]}
				
				if bgl_scaling[p]['min'] < pretrain_bgl_scaling['min']:
					pretrain_bgl_scaling['min'] = bgl_scaling[p]['min']
					
				if bgl_scaling[p]['max'] > pretrain_bgl_scaling['max']:
					pretrain_bgl_scaling['max'] = bgl_scaling[p]['max']
				
				if meal_scaling[p]['min'] < pretrain_meal_scaling['min']:
					pretrain_meal_scaling['min'] = meal_scaling[p]['min']
					
				if meal_scaling[p]['max'] > pretrain_meal_scaling['max']:
					pretrain_meal_scaling['max'] = meal_scaling[p]['max']
					
				if insulin_scaling[p]['min'] < pretrain_insulin_scaling['min']:
					pretrain_insulin_scaling['min'] = insulin_scaling[p]['min']
					
				if insulin_scaling[p]['max'] > pretrain_insulin_scaling['max']:
					pretrain_insulin_scaling['max'] = insulin_scaling[p]['max']
			
			for p in patients:	
				for i in range(len(pretrain_maxs)):
					if maxs[p][i] > pretrain_maxs[i]:
						pretrain_maxs[i] = maxs[p][i]
					if mins[p][i] < pretrain_mins[i]:
						pretrain_mins[i] = mins[p][i]
					
			if combined:
				validation_data, timeStampsValidation, _ = load_from_xml(zip(patients, files['validation']), time_res,
					                                       ['bgl', 'meal', 'bolus', 'basal'], verbose=False)
				                                       
			for p in patients:
				if combined:
					data_for_averages = np.append(dataRaw[p], validation_data[p], axis=0)
					timestamps_for_averages = np.append(timeStamps[p], timeStampsValidation[p])
				else:
					data_for_averages = dataRaw[p]
					timestamps_for_averages = timeStamps[p]
					
				event_data = []
				events = data_for_averages[:, index[event_type]]
				times = timestamps_for_averages
				num_events = 0
				num_events_by_time = {}
				for t in time_frames:
					num_events_by_time[t[1]] = 0
				for i in range(len(events)):
					if events[i] != 0.0:
						event_data.append((events[i], times[i]))
				avgs[p] = 0
				avgs_by_time[p] = {}
				for t in time_frames:
					avgs_by_time[p][t[1]] = 0
				for e in event_data:
					avgs[p] += e[0]
					num_events += 1
					hour = e[1].hour
					for t in time_frames:
						if hour >= t[0][0] and hour < t[0][1]:
							avgs_by_time[p][t[1]] += e[0]
							num_events_by_time[t[1]] += 1
							break
					
				avgs[p] /= num_events
				for t in time_frames:
					if num_events_by_time[t[1]] == 0:
						avgs_by_time[p][t[1]] = 0
					else:
						avgs_by_time[p][t[1]] /= num_events_by_time[t[1]]
						
	information = {
		'stats': stats,
		'scaling': {'bgl': bgl_scaling, 'meal': meal_scaling, 'insulin': insulin_scaling},
		'pretrain_scaling': {'bgl': pretrain_bgl_scaling, 'meal': pretrain_meal_scaling, 'insulin': pretrain_insulin_scaling},
		'maxs': maxs,
		'mins': mins,
		'pretrain_maxs': pretrain_maxs,
		'pretrain_mins': pretrain_mins,
		'avgs': avgs,
		'avgs_by_time': avgs_by_time
	}
	
	if save:
		f_name = 'stats_' + event_type + '.pkl'
		fd = open(f_name, 'wb')
		pickle.dump(information, fd)
		fd.close()
	
	return information

# This function is responsible for pre-processing the data. The data will remove specific outliers, 
# move meals to their correct position relative to their associated bolus, and add meals that presumably
# should be there but were likely not logged by the subject
#
# Parameters:
#	patients = list of subjects
#
#	raw_data = The raw data loaded from the XML files
#
#	meal_bolus_distance = the desired time in minutes between a meal and it's associated 
#	bolus. This should likely not be more than 15.
#
#	time_res = time resolution in minutes. Likely should stay at 5 minutes
#
#	use_bolus_carbs = True if you want to replace the carb counts from the smartphone
#	interface (the carbs field of the data) with the carb counts input to the bolus
#	wizard for the bolus associated with the meal. False if you want to keep the carb
#	counts as they are.
#
# Returns:
#	Data in the same format as before, but with the pre-processing adjustments made.

def pre_process_data(patients, raw_data, meal_bolus_distance=10, time_res=5, use_bolus_carbs=False):
	if meal_bolus_distance != None:
		offset = int(meal_bolus_distance / time_res)
	for p in patients:	
		for i in range(len(raw_data[p])):
			if raw_data[p][i][index['meal']] == 450:
				raw_data[p][i][index['meal']] = 0
		
			if meal_bolus_distance != None:
				if raw_data[p][i][index['bolus']] != 0 and raw_data[p][i][index['bolus_carbs']] != 0 and (raw_data[p][i][index['bolus_type']] == bolus_type_values['normal'] or raw_data[p][i][index['bolus_type']] == bolus_type_values['normal dual']):
					meal_value = raw_data[p][i][index['bolus_carbs']]
					carb_value = 0
					
					event_time = i
					closest_meal_found = False
					closest_meal = None
					left_index = i
					right_index = i
					
					while not closest_meal_found:
						if left_index < 0 or right_index >= len(raw_data[p]):
							break
						else:
							if raw_data[p][left_index][index['meal']] != 0 and raw_data[p][right_index][index['meal']] != 0:
								m1 = raw_data[p][left_index][index['meal']]
								m2 = raw_data[p][right_index][index['meal']]
								if abs(m1 - meal_value) < abs(m2 - meal_value):
									carb_value = m1
									closest_meal = left_index
								else:
									carb_value = m2
									closest_meal = right_index
								closest_meal_found = True
							elif raw_data[p][left_index][index['meal']] != 0:
								carb_value = raw_data[p][left_index][index['meal']]
								closest_meal = left_index
								closest_meal_found = True
							elif raw_data[p][right_index][index['meal']] != 0:
								carb_value = raw_data[p][right_index][index['meal']]
								closest_meal = right_index
								closest_meal_found = True
							left_index -= 1
							right_index += 1

					if not use_bolus_carbs:
						meal_value = carb_value
						
					if closest_meal_found:
						raw_data[p][closest_meal][index['meal']] = 0
						raw_data[p][event_time + offset][index['meal']] = meal_value
					
	
	return raw_data

# This is the function that actually takes the raw data and generates all examples
# for a specific event in the data.
#
# Parameters:
#	scaled_data = The raw data that may or may not have been pre-processed. The data
#	is assumed to have already been scaled.
#
#	pretrain_data = Same as above except this is the pretraining data and not the
#	subject specfic data.
#
#	time_stamps = The timestamps returned from the load_from_xml function
#
#	scaling = a dictonary with the scaling parameters for the label of the form {'max': X, 'min': Y}
#
#	pretrain_scaling = same as above except for the pretraining data
#
#	possible times = list of potential prediction horizon in TIME STEPS (not minutes)
#	to convert from minutes to time steps simply do this x = int(x / time_res)
#
#	event_start = the time step of the event.
#
#	avgs = The time of day averages in the form {'00-06': X}. The time frames are
#	found at the top of this file.
#
#	bgl_mask = The bgl mask returned from the load_from_xml function
#
#	event_type = The type of event that the label is. Used to determine which stats
#	to use.
#
#	approach = This should always be 2. 1 is for the original approach, where the
#	prediction horizon was fixed.
#
#	meal_bolus_distance = The number of minutes between a bolus and it's associated
#	meal. Used for the bolus given carbs scenario to determine how far after the
#	bolus to look for a meal.
#
#	pred_horizon = This is ignored unless approach 1 is used. Approach 1 is only
# 	there for legacy reasons so it should probably not be used.
#
#	event offset = The number of minutes between the start of the prediction window
#	and the actual event.
#
#	pre_event_sequence = The length in minutes of the history just prior to the
#	meal that is included for context
#
#	time_res = Time resolution in minutes.
#
#	combo = True only for bolus given carbs scenario
#
#	max_time = The maximum possible prediction horizion. This is used to determine
#	how much padding should be given to examples so that LSTM 2 can process them.
#	If set to None, this is done automatically by using the values in possible_times.
#	This is likely only used if you want to train only on a specific prediction horizon,
#	but add enough padding to the examples so that they can be used by the models.
#
# Returns:
# 	A tuple (patient specific examples, pretrain examples)

def create_examples(scaled_data,
                    pretrain_data, 
                    time_stamps, 
                    scaling, 
                    pretrain_scaling,
                    possible_times, 
                    event_start, 
                    avgs,
                    bgl_mask,
                    event_type='meal',
                    approach=2, 
                    meal_bolus_distance=10,
                    pred_horizon=60, 
                    event_offset=10, 
                    pre_event_sequence=360, 
                    time_res=5,
                    combo=False, 
                   	max_time=None
                   ):
                   
	examples = []
	pretrain_examples = []
		
	if max_time is None:	
		max_time_value = max(possible_times) + int(event_offset / time_res) - 1
	else:
		max_time_value = int(max_time / time_res) + int(event_offset / time_res) - 1
	
	time_of_event = time_stamps[event_start]
	
	event = scaled_data[event_start][index[event_type]]
	pretrain_event = pretrain_data[event_start][index[event_type]]
	
	if meal_bolus_distance != None:
		distance = int(meal_bolus_distance / time_res)
	else:
		distance = 0

	for j in possible_times:
		ignore = False
		if approach == 2:
			offset = int(event_offset / time_res)
			window_start = event_start - offset
		else:
			window_start = event_start - j
			
		if window_start < 0:
			break
			
		time_value = j * time_res
		
		if approach == 2:
			window_end = window_start + offset + j
		else:
			horizon = int(pred_horizon / time_res)
			window_end = window_start + horizon
			
		if window_end >= len(scaled_data):
			break
			
		bgl_target = scaled_data[window_end][index['bgl']]
		pretrain_bgl_target = pretrain_data[window_end][index['bgl']]
		
		if not bgl_mask[window_end]:
			ignore = True
		
		cur_bgl = scaled_data[event_start][index['bgl']]
		pretrain_cur_bgl = pretrain_data[event_start][index['bgl']]
		
		if combo:
			secondary_event = scaled_data[event_start + distance][index['meal']]
			pretrain_secondary_event = scaled_data[event_start + distance][index['meal']]
		
		seq1_len = int(pre_event_sequence / time_res)
		seq1 = []
		pretrain_seq1 = []
		seq1_start = window_start - seq1_len
		
		missing_total = 0
		missing_last_hour = 0
		missing_t0 = False
		
		if seq1_start < 0:
			ignore = True
		else:
			for k in range(seq1_start, window_start + 1):
				features = scaled_data[k]
				bgl_feature = features[index['bgl']]
				insulin_feature = features[index['bolus']] + features[index['basal']]
				carbs_feature = features[index['meal']]
				
				seq1.append([bgl_feature, insulin_feature, carbs_feature])
				
				features = pretrain_data[k]
				bgl_feature = features[index['bgl']]
				insulin_feature = features[index['bolus']] + features[index['basal']]
				carbs_feature = features[index['meal']]
				
				pretrain_seq1.append([bgl_feature, insulin_feature, carbs_feature])
				
				if not bgl_mask[k]:
					missing_total += 1
					if k == window_start:
						missing_t0 = True
					if k >= window_start - 12 and k != window_start:
						missing_last_hour += 1
						
			if missing_t0 or (missing_total > 12) or (missing_last_hour > 2):
				ignore = True
	
			seq1 = np.array(seq1)
			pretrain_seq1 = np.array(pretrain_seq1)
		
		seq2 = []
		pretrain_seq2 = []
		seq2_start = window_start + 1
		
		event_before = False
		event_after = False
		
		for k in range(seq2_start, window_end):
			meal_allowed = False
			features = scaled_data[k]
			insulin_feature = features[index['bolus']]
			carbs_feature = features[index['meal']]
			if k == event_start:
				if event_type == 'meal':
					carbs_feature = 0.0
				elif event_type == 'bolus':
					insulin_feature = 0.0
				
			if (k == event_start + distance) and combo:
				meal_allowed = True
			
			if (carbs_feature != 0.0 and not meal_allowed) or insulin_feature != 0.0:
				if k > event_start:
					event_after = True
				else:
					event_before = True
				
			seq2.append([insulin_feature, carbs_feature])
			
			features = pretrain_data[k]
			insulin_feature = features[index['bolus']]
			carbs_feature = features[index['meal']]
			
			if k == event_start:
				if event_type == 'meal':
					carbs_feature = 0.0
				elif event_type == 'bolus':
					insulin_feature = 0.0
				
			pretrain_seq2.append([insulin_feature, carbs_feature])
			
		for k in range(max_time_value - len(seq2)):
			seq2.append([-1.0, -1.0])
			pretrain_seq2.append([-1.0, -1.0])

		seq2 = np.array(seq2)
		pretrain_seq2 = np.array(pretrain_seq2)
		
		case = None
		if (not event_before) and (not event_after):
			case = '1'
		elif (event_before) and (not event_after):
			case = '2'
		else:
			case = '3'
					
		hour = time_of_event.hour	
		for time in time_frames:
			if hour >= time[0][0] and hour < time[0][1]:
				avg = avgs[time[1]]
				break
		
		if not ignore:
			if combo:
				example = {'label': event,
						   'input1': seq1,
						   'input2': seq2,
						   'input3': np.array([bgl_target] + [time_value / int(max_time_value * time_res)] + [(avg - scaling['min']) / scaling['max']] + [secondary_event]),
						   'case': case
						  }
			else:
				example = {'label': event,
						   'input1': seq1,
						   'input2': seq2,
						   'input3': np.array([bgl_target] + [time_value / int(max_time_value * time_res)] + [(avg - scaling['min']) / scaling['max']]),
						   'case': case
						  }
				
			examples.append(example)
			
			if combo:
				pretrain_example = {'label': pretrain_event,
									'input1': pretrain_seq1,
									'input2': pretrain_seq2,
									'input3': np.array([pretrain_bgl_target] + [time_value / int(max_time_value * time_res)] + [(avg - pretrain_scaling['min']) / pretrain_scaling['max']] + [pretrain_secondary_event]),
									'case': case
								   }
			else:
				pretrain_example = {'label': pretrain_event,
									'input1': pretrain_seq1,
									'input2': pretrain_seq2,
									'input3': np.array([pretrain_bgl_target] + [time_value / int(max_time_value * time_res)] + [(avg - pretrain_scaling['min']) / pretrain_scaling['max']]),
									'case': case
								   }
							   
			pretrain_examples.append(pretrain_example)
		
	return (examples, pretrain_examples)

# This function takes the examples created by the previous function and formats them.
#
# Parameters:
#	patients = list of patient IDs
#
#	all_examples = a tuple (subject specific examples, pretrain examples)
#
# Returns:
#	A tuple ((feature vectors, labels), (pretrain feature vectors, pretrain labels))
#
def create_vectors(patients, all_examples):
	feature_vectors = {}
	labels = {}
	pretrain_feature_vectors = {}
	pretrain_labels = {}
	
	for d in ['training', 'testing', 'validation']:
		feature_vectors[d] = {}
		labels[d] = {}
		pretrain_feature_vectors[d] = {}
		pretrain_labels[d] = {}
		for p in patients:
			feature_vectors[d][p] = {}
			labels[d][p] = {}
			for c in ['1', '2', '3']:
				feature_vectors[d][p][c] = {'input1_layer': [],
											'input2_layer': [],
											'input3_layer': []
										   }
				labels[d][p][c] = []
		for c in ['1', '2', '3']:
			pretrain_feature_vectors[d][c] = {'input1_layer': [],
											  'input2_layer': [],
											  'input3_layer': []
											 }
			pretrain_labels[d][c] = []
	
	for d in ['training', 'testing', 'validation']:
		for p in patients:
			for ex in all_examples[0][d][p]:
				case = ex['case']
				
				feature_vectors[d][p]['3']['input1_layer'].append(ex['input1'])
				feature_vectors[d][p]['3']['input2_layer'].append(ex['input2'])
				feature_vectors[d][p]['3']['input3_layer'].append(ex['input3'])
				labels[d][p]['3'].append(ex['label'])
				
				if case == '2' or case == '1':
					
					feature_vectors[d][p]['2']['input1_layer'].append(ex['input1'])
					feature_vectors[d][p]['2']['input2_layer'].append(ex['input2'])
					feature_vectors[d][p]['2']['input3_layer'].append(ex['input3'])
					labels[d][p]['2'].append(ex['label'])
					
				if case == '1':
					
					feature_vectors[d][p]['1']['input1_layer'].append(ex['input1'])
					feature_vectors[d][p]['1']['input2_layer'].append(ex['input2'])
					feature_vectors[d][p]['1']['input3_layer'].append(ex['input3'])
					labels[d][p]['1'].append(ex['label'])
					
		for ex in all_examples[1][d]:
			case = ex['case']
			
			pretrain_feature_vectors[d]['3']['input1_layer'].append(ex['input1'])
			pretrain_feature_vectors[d]['3']['input2_layer'].append(ex['input2'])
			pretrain_feature_vectors[d]['3']['input3_layer'].append(ex['input3'])
			pretrain_labels[d]['3'].append(ex['label'])
			
			if case == '2' or case == '1':
				
				pretrain_feature_vectors[d]['2']['input1_layer'].append(ex['input1'])
				pretrain_feature_vectors[d]['2']['input2_layer'].append(ex['input2'])
				pretrain_feature_vectors[d]['2']['input3_layer'].append(ex['input3'])
				pretrain_labels[d]['2'].append(ex['label'])
				
			if case == '1':
				
				pretrain_feature_vectors[d]['1']['input1_layer'].append(ex['input1'])
				pretrain_feature_vectors[d]['1']['input2_layer'].append(ex['input2'])
				pretrain_feature_vectors[d]['1']['input3_layer'].append(ex['input3'])
				pretrain_labels[d]['1'].append(ex['label'])
					
		for p in patients:
			for c in ['1', '2', '3']:
				labels[d][p][c] = np.array(labels[d][p][c])
				for l in ['input1_layer', 'input2_layer', 'input3_layer']:
					feature_vectors[d][p][c][l] = np.array(feature_vectors[d][p][c][l])
					
		for c in ['1', '2', '3']:
			pretrain_labels[d][c] = np.array(pretrain_labels[d][c])
			for l in ['input1_layer', 'input2_layer', 'input3_layer']:
				pretrain_feature_vectors[d][c][l] = np.array(pretrain_feature_vectors[d][c][l])
	
	return ((feature_vectors, labels), (pretrain_feature_vectors, pretrain_labels))

# This is the function that performs all of the actions necessary to go from raw
# XML file to examples saved in a pickle file. Below there are functions to do this
# for the carb, bolus, and bolus given carbs recommendation scenarios.
#
# Parameters:
#	patients = list of patient IDs
#
#	files = dictonary of file names of the form {'training': ['559-training.xml', ...]}.
#	This dictonary should include file names for training, testing, and validation.
#	The order of the lists should follow the order in the patients parameter.
#	For example, if the first patient ID in patients is '559', the first filename
# 	in files['training'] should be the file name containing subject 559's training data.
#
#	stats_file = The name of the pickle file holding the stats. This is only needed if
#	the stats were calculated ahead of time.
#
#	approach = This should be 2, 1 is only there for legacy reasons.
#
#	times = Possible prediction horizions in minutes
#
#	meal_bolus_distance = The desired amount of time between a bolus and the meal that
#	it is associated with.
#
#	time_res = time resolution in minutes
#
#	pred_horizon = The fixed prediction horizion. This is only used for approach 1.
#
#	use_bolus_carbs = True if you want the pre-processing step to replace the carb
#	counts from the smartphone interface with the carb input to the bolus wizard.
#
#	combined = True if training on the union of training and validation data.
#
#	max_time = The maximum prediction horizon. This is only needed if you want
#	to make sure that the sequences are padded properly. The only reason they wouldn't
#	be would be in a case where you are generating examples with a specific prediction
#	horizon, and using a model that expects sequences longer than those generated with
#	that prediction horizon.

def load_carbs_data(patients, 
                    files, 
                    stats_file = None,
                    approach=2,
                    times=['30','35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90'],
                    meal_bolus_distance=10, 
                    time_res=5, 
                    pred_horizon=60,
                    use_bolus_carbs=False, 
                    combined=False,
                    max_time=None
                   ):

	if stats_file == None:           
		information = calculate_stats(patients, 
		                              files, 
		                              event_type='meal', 
		                              time_res=time_res, 
		                              combined=combined
		                             )
	else:
		fd = open(stats_file, 'rb')
		information = pickle.load(fd)
		fd.close()
		
	possible_times = []
	for t in times:
		possible_times.append(int(t / time_res)) 
	
	maxs = information['maxs']
	mins = information['mins']
	
	pretrain_maxs = information['pretrain_maxs']
	pretrain_mins = information['pretrain_mins']
	
	scaling = information['scaling']['meal']
	pretrain_scaling =information['pretrain_scaling']['meal']
	
	all_examples = {}
	all_pretrain_examples = {}
	
	for d in ['training', 'testing', 'validation']:
		raw_data, time_stamps, bgl_mask = load_from_xml(zip(patients, files[d]), 
		                                                       time_res, 
		                                                       ['bgl', 'meal', 'bolus', 'basal'],
		                                                        verbose=False, 
		                                                       )
		                                                       
		raw_data = pre_process_data(patients, raw_data, meal_bolus_distance=meal_bolus_distance, time_res=time_res, use_bolus_carbs=use_bolus_carbs)                                        
		
		all_examples[d] = {}
		all_pretrain_examples[d] = []
		
		scaled_data = {}
		pretrain_data = {}
		for p in patients:
			all_examples[d][p] = []
		
			scaled_data[p] = (raw_data[p] - mins[p]) / maxs[p]
			pretrain_data[p] = (raw_data[p] - pretrain_mins) / pretrain_maxs
	
		if approach == 2:
			event_offset = 10
		else:
			event_offset = 0
			
		for p in patients:
			
			avgs = information['avgs_by_time'][p]

			for i in range(len(scaled_data[p])):
				#if scaled_data[p][i][index['meal']] != 0:
				if scaled_data[p][i][index['meal']] != 0 and scaled_data[p][i - 2][index['bolus']] == 0:
					event_start = i
					examples = create_examples(scaled_data[p], 
					                           pretrain_data[p],
							                   time_stamps[p], 
							                   scaling[p],
							                   pretrain_scaling,
							                   possible_times, 
							                   event_start, 
							                   avgs,
							                   bgl_mask[p],
							                   event_type='meal', 
							                   approach=approach, 
							                   meal_bolus_distance=meal_bolus_distance,
							                   pred_horizon=pred_horizon, 
							                   event_offset=event_offset,
							                   combo=False,
							                   max_time=max_time
							                  )
							                  
					(ex, p_ex) = examples
					
					all_examples[d][p] += ex
					all_pretrain_examples[d] += p_ex
					
	vectors = create_vectors(patients, (all_examples, all_pretrain_examples))
	((v, l), (p_v, p_l)) = vectors
	
	info = {
		'scenario': 'carbs',
		'possible_horizons': possible_times,
		'approach': approach,
		'meal_bolus_distance': meal_bolus_distance,
		'event_offset': event_offset
	}
	
	data = {}
	pretrain_data = {}
	for d in ['training', 'testing', 'validation']:
		data[d] = (v[d], l[d])
		pretrain_data[d] = (p_v[d], p_l[d])
		
	data['scaling'] = scaling
	data['averages'] = information['avgs']
	data['averages_by_time'] = information['avgs_by_time']
	
	pretrain_data['scaling'] = pretrain_scaling
	
	return ((data, pretrain_data), info)
	
def load_bolus_data(patients, 
                    files, 
                    stats_file = None,
                    approach=2,
                    times=['30','35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90'],
                    meal_bolus_distance=10,
                    time_res=5, 
                    pred_horizon=60, 
                    use_bolus_carbs=False,
                    combined=False,
                    max_time=None
                   ):

	if stats_file == None:           
		information = calculate_stats(patients, 
		                              files, 
		                              event_type='bolus', 
		                              time_res=time_res, 
		                              combined=combined
		                             )
	else:
		fd = open(stats_file, 'rb')
		information = pickle.load(fd)
		fd.close()
	
	possible_times = []
	for t in times:
		possible_times.append(int(t / time_res)) 
	
	maxs = information['maxs']
	mins = information['mins']
	
	pretrain_maxs = information['pretrain_maxs']
	pretrain_mins = information['pretrain_mins']
	
	scaling = information['scaling']['insulin']
	pretrain_scaling =information['pretrain_scaling']['insulin']
	
	all_examples = {}
	all_pretrain_examples = {}
	
	for d in ['training', 'testing', 'validation']:
		raw_data, time_stamps, bgl_mask = load_from_xml(zip(patients, files[d]), 
		                                                       time_res, 
		                                                       ['bgl', 'meal', 'bolus', 'basal'],
		                                                        verbose=False, 
		                                                       )
		
		raw_data = pre_process_data(patients, raw_data, meal_bolus_distance=meal_bolus_distance, time_res=time_res, use_bolus_carbs=use_bolus_carbs)
		                                                       
		all_examples[d] = {}
		all_pretrain_examples[d] = []
		
		scaled_data = {}
		pretrain_data = {}
		for p in patients:
			all_examples[d][p] = []
		
			scaled_data[p] = (raw_data[p] - mins[p]) / maxs[p]
			pretrain_data[p] = (raw_data[p] - pretrain_mins) / pretrain_maxs
	
		if approach == 2:
			event_offset = 10
		else:
			event_offset = 0
			
		for p in patients:
			
			avgs = information['avgs_by_time'][p]
			
			for i in range(len(scaled_data[p])):
				if scaled_data[p][i][index['bolus']] != 0 and (scaled_data[p][i][index['bolus_type']] == bolus_type_values['normal'] or scaled_data[p][i][index['bolus_type']] == bolus_type_values['normal dual']):
					event_start = i
					examples = create_examples(scaled_data[p], 
					                           pretrain_data[p],
							                   time_stamps[p], 
							                   scaling[p],
							                   pretrain_scaling,
							                   possible_times, 
							                   event_start, 
							                   avgs,
							                   bgl_mask[p],
							                   event_type='bolus', 
							                   approach=approach, 
							                   meal_bolus_distance=meal_bolus_distance,
							                   pred_horizon=pred_horizon, 
							                   event_offset=event_offset,
							                   combo=False,
							                   max_time=max_time
							                  )
							                  
					(ex, p_ex) = examples
					
					all_examples[d][p] += ex
					all_pretrain_examples[d] += p_ex
					
	vectors = create_vectors(patients, (all_examples, all_pretrain_examples))
	((v, l), (p_v, p_l)) = vectors
	
	info = {
		'scenario': 'bolus',
		'possible_horizons': possible_times,
		'approach': approach,
		'meal_bolus_distance': meal_bolus_distance,
		'event_offset': event_offset
	}
	
	data = {}
	pretrain_data = {}
	for d in ['training', 'testing', 'validation']:
		data[d] = (v[d], l[d])
		pretrain_data[d] = (p_v[d], p_l[d])
		
	data['scaling'] = scaling
	data['averages'] = information['avgs']
	data['averages_by_time'] = information['avgs_by_time']
	
	pretrain_data['scaling'] = pretrain_scaling
	
	return ((data, pretrain_data), info)
	
def load_combo_data(patients, 
                    files, 
                    stats_file = None,
                    approach=2, 
                    times=['30','35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90'],
                    meal_bolus_distance=10,
                    time_res=5, 
                    pred_horizon=60, 
                    use_bolus_carbs=False,
                    combined=False,
                    max_time=None
                   ):
                  
	info = {}
     
	distance = int(meal_bolus_distance / time_res)

	if stats_file == None:           
		information = calculate_stats(patients, 
		                              files, 
		                              event_type='bolus', 
		                              time_res=time_res, 
		                              combined=combined
		                             )
	else:
		fd = open(stats_file, 'rb')
		information = pickle.load(fd)
		fd.close()
		
	possible_times = []
	for t in times:
		possible_times.append(int(t / time_res)) 
	
	maxs = information['maxs']
	mins = information['mins']
	
	pretrain_maxs = information['pretrain_maxs']
	pretrain_mins = information['pretrain_mins']
	
	scaling = information['scaling']['insulin']
	pretrain_scaling =information['pretrain_scaling']['insulin']
	
	all_examples = {}
	all_pretrain_examples = {}
	
	for d in ['training', 'testing', 'validation']:
		raw_data, time_stamps, bgl_mask = load_from_xml(zip(patients, files[d]), 
		                                                       time_res, 
		                                                       ['bgl', 'meal', 'bolus', 'basal'],
		                                                        verbose=False, 
		                                                       )
		                                                       
		raw_data = pre_process_data(patients, raw_data, meal_bolus_distance=meal_bolus_distance, time_res=time_res, use_bolus_carbs=use_bolus_carbs)
		                                                       
		all_examples[d] = {}
		all_pretrain_examples[d] = []
		
		scaled_data = {}
		pretrain_data = {}
		for p in patients:
			all_examples[d][p] = []
		
			scaled_data[p] = (raw_data[p] - mins[p]) / maxs[p]
			pretrain_data[p] = (raw_data[p] - pretrain_mins) / pretrain_maxs
	
		if approach == 2:
			event_offset = 10
		else:
			event_offset = 0
			
		for p in patients:
			avgs = information['avgs_by_time'][p]
			for i in range(len(scaled_data[p])):
				if (scaled_data[p][i][index['bolus']] != 0 and scaled_data[p][i][index['bolus_carbs']] != 0) and (scaled_data[p][i][index['bolus_type']] == bolus_type_values['normal'] or scaled_data[p][i][index['bolus_type']] == bolus_type_values['normal dual']):
					if i + distance < len(scaled_data[p]) and scaled_data[p][i + distance][index['meal']] != 0:
						event_start = i
						examples = create_examples(scaled_data[p], 
							                       pretrain_data[p],
									               time_stamps[p], 
									               scaling[p],
									               pretrain_scaling,
									               possible_times, 
									               event_start, 
									               avgs,
									               bgl_mask[p],
									               event_type='bolus', 
									               approach=approach, 
									               meal_bolus_distance=meal_bolus_distance,
									               pred_horizon=pred_horizon, 
									               event_offset=event_offset,
									               combo=True,
									               max_time=max_time
									              )
									              
						(ex, p_ex) = examples
						
						all_examples[d][p] += ex
						all_pretrain_examples[d] += p_ex

	vectors = create_vectors(patients, (all_examples, all_pretrain_examples))
	((v, l), (p_v, p_l)) = vectors
	
	info = {
		'scenario': 'combo',
		'possible_horizons': possible_times,
		'approach': approach,
		'meal_bolus_distance': meal_bolus_distance,
		'event_offset': event_offset
	}
	
	data = {}
	pretrain_data = {}
	for d in ['training', 'testing', 'validation']:
		data[d] = (v[d], l[d])
		pretrain_data[d] = (p_v[d], p_l[d])
		
	data['scaling'] = scaling
	data['averages'] = information['avgs']
	data['averages_by_time'] = information['avgs_by_time']
	
	pretrain_data['scaling'] = pretrain_scaling
	
	return ((data, pretrain_data), info)
	
if __name__ == '__main__':

	patients = ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']

	training_files = ['/home/jeremy/EventPrediction/data/data/training/' + pID + '-ws-training-new.xml' for pID in patients]
	testing_files = ['/home/jeremy/EventPrediction/data/data/testing/' + pID + '-ws-testing.xml' for pID in patients]
	validation_files = ['/home/jeremy/EventPrediction/data/data/validation/' + pID + '-ws-validation.xml' for pID in patients]
	files = {'training': training_files, 'testing': testing_files, 'validation': validation_files}


	stats_loaded = False
	carbs = True
	bolus = False
	combo = False

	combined = False
	bolus_carbs = True
	
	if carbs and not stats_loaded:
		meal_stats = calculate_stats(patients, 
				                     files, 
				                     event_type='meal', 
				                     time_res=5, 
				                     meal_bolus_distance=10,
				                     use_bolus_carbs=bolus_carbs,
				                     combined=combined,
				                     save=True
				                    )
	if (bolus or combo) and not stats_loaded:                 
		bolus_stats = calculate_stats(patients, 
				                      files, 
				                      event_type='bolus', 
				                      time_res=5, 
				                      meal_bolus_distance=10,
				                      use_bolus_carbs=bolus_carbs,
				                      combined=combined, 
				                      save=True
				                     )
				                     
	if carbs:  
		carbs_data, carbs_info = load_carbs_data(patients, 
				                     files, 
				                     stats_file='stats_meal.pkl',
				                     approach=2, 
				                     times=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
				                     meal_bolus_distance=10,
				                     time_res=5,
				                     pred_horizon=60,
				                     use_bolus_carbs=bolus_carbs,
				                     combined=combined
				                    )
	
	if bolus:	                       
		bolus_data, bolus_info = load_bolus_data(patients, 
				                     files, 
				                     stats_file='stats_bolus.pkl',
				                     approach=2,
				                     times=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
				                     meal_bolus_distance=10,
				                     time_res=5, 
				                     pred_horizon=60, 
				                     use_bolus_carbs=bolus_carbs,
				                     combined=combined
				                    )
	if combo:                    
		combo_data, combo_info = load_combo_data(patients, 
				                     files, 
				                     stats_file='stats_bolus.pkl',
				                     approach=2, 
				                     times=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
				                     meal_bolus_distance=10,
				                     time_res=5,
				                     pred_horizon=60, 
				                     use_bolus_carbs=bolus_carbs,
				                     combined=combined
				                    )
      
	info = {}       
	if carbs:
		fd = open('carbs_data.pkl', 'wb')
		pickle.dump(carbs_data, fd)
		fd.close()
		info['carbs'] = carbs_info

	if bolus:
		fd = open('bolus_data.pkl', 'wb')
		pickle.dump(bolus_data, fd)
		fd.close()
		info['bolus'] = bolus_info

	if combo:
		fd = open('combo_data.pkl', 'wb')
		pickle.dump(combo_data, fd)
		fd.close()
		info['combo'] = combo_info
		
	if info != {}:	
		fd = open('data_info.json', 'w')
		json.dump(info, fd)
		fd.close()
	
