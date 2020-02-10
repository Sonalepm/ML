##################################################
###   Decision Tree stump 
##################################################

import sys

datafile = sys.argv[1]
labelfile = sys.argv[2]

########################################
### Read data
########################################

f = open(datafile)
line = f.readline()
data = []
while(line != ''):
	a = line.split()
	l = []
	for j in range(len(a)):
		l.append(float(a[j]))
#	l.append(float(1))
	data.append(l)
	line = f.readline()
rows = len(data)
cols = len(data[0])
f.close()
#print(rows,cols)
#print(data)

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

dataset =[]
for i in range(len(data)):
	if (trainlabels.get(i) != None):
		data[i].append(trainlabels[i])	
		dataset.append(data[i])

#print(dataset)

####################################
### Splitting dataset  
####################################
def split_ds(dataset,value,col):
	leftds = []
	rightds = []
	for row in dataset:
		if (row[col] < value):
			leftds.append(row)
		else:
			rightds.append(row)
	return leftds,rightds


###################################
###	Find gini index
##(lsize/rows)*(lp/lsize)*(1 - lp/lsize) + (rsize/rows)*(rp/rsize)*(1 - rp/rsize)
###################################
def find_gini_index(group,class_val):
	gini_index = 0
	rows = len(group[0])+ len(group[1])
	for g in group: 
		if len(g) == 0:
			continue
		pa = 1
		for class_  in class_val: 
			p = [row[-1] for row in g].count(class_)/(len(g[0]))
			pa *= p
		gini_index += (len(g[0])/rows) * pa
	return gini_index


#####################################
### Get best split
#####################################

def get_split(dataset):
	class_val = list(set(row[-1] for row in dataset))
	#print(class_val)
	best_col,best_row,best_value,best_gini,best_score,best_group = 0,0,0,1,0,None
	count = 0
	#Loop through each column ,find val that gives min gini index
	for j in range(len(dataset[0])-1):
		for i in range(len(dataset)):
			group = split_ds(dataset,dataset[i][j],j)
			gini = find_gini_index(group,class_val)
	#		print(gini)
	#		print("Best gini:",best_gini)
			if gini < best_gini:
				best_col,best_row,best_value,best_gini,best_group = j,i,dataset[i][j],gini,group
			elif gini == best_gini:
				count += 1

	#print("Gini < bestGIni",best_col,best_row,best_value,best_gini,best_group)
	#print("Count:",count)	
	if (count == ((len(dataset) * 2) - 1)):
		best_col,best_row,best_rowv = 0,0,dataset[0][best_col]
		#print("best_Col",best_col)
		#print("best_row",best_row)
		#print("best row value",best_rowv)
		
		for i in range(len(dataset)):
			if dataset[i][best_col] > best_rowv :
				best_row,best_rowv = i,dataset[i][best_col]
		best_value,best_gini = dataset[best_row][best_col],gini
		best_group = split_ds(dataset,best_value,best_col)
	#print('column:',best_col,',row',best_row,',value',best_value,',gini',best_gini)
	#print("Best row",best_row)	
	return best_col,best_value,best_gini


def get_splitvalue(best_col,best_value,dataset):
	top_col = []
	upperval = -1000000
	for i in range(len(dataset)):
		value = dataset[i][best_col]
		if value < best_value:
			if value > upperval:
				upperval = value
	value = (upperval + best_value)/2
	return value 


best_col,best_val,best_gini = get_split(dataset)
val = get_splitvalue(best_col,best_val,dataset)

print("Best Column:",best_col)
#print("Gini Value:",best_gini)
print("Split value:",val)



