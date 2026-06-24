################################################################################
# Split Data
# Written by: Jeremy Beauchamp
# Description: Finds the amount of examples in last 10, previous 10, and the rest
# 			   of the days.
################################################################################

from lxml import etree as ET
import argparse
import datetime

parser = argparse.ArgumentParser(description='Determine which files to check')
parser.add_argument('--files', type=str, help='a list of files that should be checked',
					default='540-ws-training.xml 544-ws-training.xml 552-ws-training.xml 559-ws-training.xml 563-ws-training.xml 567-ws-training.xml 570-ws-training.xml 575-ws-training.xml 584-ws-training.xml 588-ws-training.xml 591-ws-training.xml 596-ws-training.xml')
					
args = parser.parse_args()
files = args.files.split()
pids = []
for f in files:
	pids.append(f[:3])
	
ts_events = ['glucose_level',
			 'finger_stick',
			 'basal',
			 'meal',
			 'stressors',
			 'hypo_event',
			 'exercise',
			 'basis_heart_rate',
			 'basis_gsr',
			 'basis_skin_temperature',
			 'basis_air_temperature',
			 'basis_steps',
		    ]
	
for f in files:
	tree = ET.parse(f)
	pid = f[:3]
	root = tree.getroot()
	for child in root:
		if(child.tag == 'meal'):
			meals = child
			break
	
	first_day = meals[0].attrib['ts']
	for meal in meals:
		last_day = meal.attrib['ts']
		
	first_day = datetime.datetime.strptime(first_day, "%d-%m-%Y %H:%M:%S")
	last_day = datetime.datetime.strptime(last_day, "%d-%m-%Y %H:%M:%S")
	
	cut_off = last_day - datetime.timedelta(days=10)
	
	patient = ET.Element(root.tag)
	for i in root.attrib.keys():
		patient.set(i, root.attrib[i])
		
	validation_patient = ET.Element(root.tag)
	for i in root.attrib.keys():
		validation_patient.set(i, root.attrib[i])

	for child in root:
		element = ET.Element(child.tag)
		validation_element = ET.Element(child.tag)
		for item in child:
			if child.tag in ts_events:
				time = datetime.datetime.strptime(item.attrib['ts'], "%d-%m-%Y %H:%M:%S")
			elif child.tag == 'basis_sleep':
				time = datetime.datetime.strptime(item.attrib['tbegin'], "%d-%m-%Y %H:%M:%S")
			else:
				time = datetime.datetime.strptime(item.attrib['ts_begin'], "%d-%m-%Y %H:%M:%S")
			if time >= cut_off:
				sub_element = ET.Element(item.tag)
				for i in item.attrib.keys():
					sub_element.set(i, item.attrib[i])
				validation_element.append(sub_element)
			else:
				sub_element = ET.Element(item.tag)
				for i in item.attrib.keys():
					sub_element.set(i, item.attrib[i])
				element.append(sub_element)
					
		patient.append(element)
		validation_patient.append(validation_element)
	
	tree = ET.ElementTree(patient)
	validation_tree = ET.ElementTree(validation_patient)
	tree.write(pid + '-ws-training-new.xml', pretty_print=True)
	validation_tree.write(pid + '-ws-validation.xml', pretty_print=True)
					
