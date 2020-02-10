import sys

datafile = sys.argv[1]
f = open(datafile)
data = []
i = 0
l = f.readline()

###############
### Read data
###############
while(l != ''):
	a = l.split()
	l2 = []
	for j in range(0,len(a), 1):
		l2.append(float(a[j]))
	data.append(l2)
	l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()

##############
### Read labels
##############
labelfile = sys.argv[2]
f = open(labelfile)
trainlabels = {}
n = []
n.append(0)
n.append(0)
l = f.readline()
while(l != ''):
	a = l.split()
	trainlabels[int(a[1])] = int (a[0])
	l =f.readline()
	n[int(a[0])] += 1

#################
#### Compute means
#################
m0 = []
for j in range(0,cols,1):
	m0.append(0)
m1 = []
for j in range(0,cols,1):
	m1.append(0)

for i in range(0,rows,1):
	if(trainlabels.get(i) != None and trainlabels[i] == 0):
		for j in range(0,cols,1):
			m0[j] = m0[j] + data[i][j]
	if(trainlabels.get(i) != None and trainlabels[i] == 1):
		for j in range(0,cols,1):
			m1[j] = m1[j] + data[i][j]

for j in range(0,cols,1):
	m0[j] = m0[j]/n[0]
	m1[j] = m1[j]/n[1]

#print(m0)
#print(m1)

############
#### Classify unlabeled points
############

for i in range(0,rows,1):
	if(trainlabels.get(i) == None):
		d0 = 0
		d1 = 0 
		for j in range(0,cols,1):
			d0 = d0 + ( m0[j] - data[i][j])**2
			d1 = d1 + ( m1[j] - data[i][j])**2
		if (d0<d1):
			print("0 ",i)
		else:
			print("1 ",i)

