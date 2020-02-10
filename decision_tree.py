############################################
###	Bagging decision stump
############################################

import sys
import math
import random

datafile = sys.argv[1]
labelfile = sys.argv[2]

#####################################
### Read data
#####################################

f = open(datafile)
line = f.readline()
data = []
while (line != ''):
	a = line.split()
	l = []
	for i in range(len(a)):
		l.append(float(a[i]))
	data.append(l)
	line = f.readline()
rows = len(data)
cols = len(data[0])
f.close()
#print(rows,cols)

######################################
### Read labels
######################################

f = open(labelfile)
trainlabels = {}
l = f.readline()
while(l != ''):
	a = l.split()
	trainlabels[int(a[1])] = int(a[0])
	l = f.readline()
f.close()
#print(trainlabels)

#####################################
### Adding labels to dataset
#####################################

ds = []
for i in range(len(data)):
	if(trainlabels.get(i) != None):
		data[i].append(trainlabels[i])
		ds.append(data[i])
#print(ds[0])

####################################
### create boostrapped dataset
####################################
def bootstrap_ds(dataset):
	dset = []
	dat = []
	ds_len = round(0.9 * len(dataset))
	#print("length",ds_len)
	for i in range(ds_len):
		dat.append(0)
		dat[i] = random.randint(0,ds_len)
	for j in dat:
		dset.append(dataset[j])
	return dset

####################################
### test split based on attribute and val
####################################
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def gini_index(groups,classes):
	n = float(sum([len(group) for group in groups]))
	gini_ind = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue 	# to avoid divide by zero
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		gini_ind += (1.0 - score) * (size / n)
	return gini_ind

#####################################
### get split
#####################################
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index,row[index], gini, groups
	return {'index':b_index,'value':b_value,'groups':b_groups,'gini':b_score}


def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'],max_depth,min_size,depth+1)

	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)


###################################
### Build a decision tree and print values
###################################
def get_tree(traindata,max_depth,min_size):
	root = get_split(traindata)
	split(root,max_depth,min_size,1)
	node = root
	return {'index':node['index'], 'value':node['value'], 'left':node['left'], 'right':node['right'], 'gini':node['gini']}		

predicted_list = {}
repeat = 0

####################################
### Get predicted values
####################################
def predict(stump,row_data):
	if row_data[stump['index']] < stump['value']:
		if isinstance(stump['left'],dict):
			return predict(stump['left'],row_data)
		else:
			return stump['left']
	else:
		if isinstance(stump['right'],dict):
			return predict(stump['right'],row_data)
		else:
			return stump['right']

	
while (repeat < 1):
	bsdataset = bootstrap_ds(ds)
	#print(bsdataset[0])
	pt = get_tree(bsdataset,1,1)
	stump = {'index': pt['index'], 'right': pt['right'], 'value': pt['value'], 'left': pt['left'], 'gini': pt['gini']}
	#print(stump)

	pred = {}  # dict with val - predicted values 
	for i in range(len(data)):
		if (trainlabels.get(i) == None):
			prediction = predict(stump,data[i])
			pred[i] = int(prediction)
	for k,v in pred.items():
		if (repeat == 0):
			predicted_list[k] = int(v)
		else :
			predicted_list[k] += int(v)
	
	repeat += 1

for key,value in predicted_list.items():
	if (value > 1):
		value = 1
	elif (value < 1):
		value = 0
	print(value,key)

