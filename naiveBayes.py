#############################################
### Naive Bayes algorithm
#############################################

import sys
from math import sqrt

datafile = sys.argv[1]
f = open(datafile)
data = []
i = 0
l = f.readline()

#############################################
### Read data
#############################################
while(l != ''):
	a = l.split()
	l2 = []
	for j in range(0,len(a),1):
		l2.append(float(a[j]))
	data.append(l2)
	l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()

############################################
### Read labels
############################################
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
	a = l.split()
	trainlabels[int(a[1])] = int(a[0])
	l =f.readline()
	n[int(a[0])] += 1


###########################################
### Compute means and standard deviation
###########################################
mean0 = []
mean1 = []
for j in range(0,cols,1):
	mean0.append(0.01)           # Mean initialized to 0.01
	mean1.append(0.01)           # Mean initialized to 0.01
	
std0 = []
std1 = []
for j in range(0,cols,1):
	std0.append(0)
	std1.append(0)

for i in range(0,rows,1):
	if(trainlabels.get(i) != None and trainlabels[i] == 0):
		for j in range(0,cols,1):
			mean0[j] += data[i][j]
	if(trainlabels.get(i) != None and trainlabels[i] == 1):
		for j in range(0,cols,1):
			mean1[j] += data[i][j]

for j in range(0,cols,1):
	mean0[j] = mean0[j]/n[0]
	mean1[j] = mean1[j]/n[1]

for i in range(0,rows,1):
        if(trainlabels.get(i) != None and trainlabels[i] == 0):
               for j in range(0,cols,1):
                      std0[j] += (mean0[j] - data[i][j])**2 
        if(trainlabels.get(i) != None and trainlabels[i] == 1):
               for j in range(0,cols,1):
                      std1[j] += (mean1[j] - data[i][j])**2
	
for j in range(0,cols,1):
	if(std0[j] != 0):
		std0[j] = sqrt(std0[j]/n[0])
	else:
		std0[j] = 0.001 	# To avoid divide by zero error incase standard deviation becomes zero
        
	if(std1[j] != 0):
		std1[j] = sqrt(std1[j]/n[1])
	else:
		std1[j] = 0.001         # To avoid divide by zero error incase standard deviation becomes zero

############################################
### Classify unlabeled points
############################################

for i in range(0,rows,1):
	if(trainlabels.get(i) == None):
		dist0 = 0
		dist1 = 0 
		for j in range(0,cols,1):
			dist0 += (( mean0[j] - data[i][j]) / std0[j])**2
			dist1 += (( mean1[j] - data[i][j]) / std1[j])**2
		if(dist0 < dist1):
			print("0 ",i)   # Label 0
		else:
			print("1 ",i)   # Label 1

