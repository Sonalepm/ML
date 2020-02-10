##################################################
###	Gradient Descent Algorithm
###	Optimizing least squares - adaptive eta
##################################################

import sys
import random
import math

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
	l2 = []
	for j in range(0,len(a),1):
		l2.append(float(a[j]))
	l2.append(float(1))
	data.append(l2)
	line = f.readline()
rows = len(data)
cols = len(data[0])
f.close()

######################################
### Read labels
######################################

f = open(labelfile)
trainlabels = {}
l = f.readline()
while(l != ''):
	a = l.split()
	trainlabels[int(a[1])] = int(a[0])
	if ((trainlabels[int(a[1])])== 0):
		trainlabels[int(a[1])] = -1
	l =f.readline()
f.close()

#####################################
### Initialize random w
#####################################
w = []
for i in range(cols):
	w.append(0) 
for i in range(cols):
	# random number in range(-0.01,0.01)
	w[i] += (0.02*random.random() - 0.01)
	#print("W",w[i])	

######################################
### Calculating dot product
######################################

def w_transpose_x(w,data):
	y = 0.0 
	for j in range(0,cols,1):
		y += float(w[j]*data[j])
	#print(y)
	return(y)

stop_cond = 0.001 
eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
prev_error = float('inf') 
cond = True
max_iter = 0
best_eta = 1

while(cond == True):
	del_w = []
	for j in range(0,cols,1):
		del_w.append(0)

	for t in range(0,rows,1):
		y = 0.0
		if(trainlabels.get(t) != None):
			y = w_transpose_x(w,data[t])
			for j in range(0,cols,1):
				del_w[j] += (trainlabels[t] - y)* data[t][j]
	bestobj = 1000000000000  
	for k in range(len(eta_list)):
		eta = eta_list[k]
		for j in range(cols):
			w[j] += eta * del_w[j]
		error = 0.0
		for t in range(rows):
			if(trainlabels.get(t) != None):
				temp = (trainlabels[t]-w_transpose_x(w,data[t]))
				error += float(temp**2)
		obj = error
		if(obj<bestobj):
			best_eta = eta
			bestobj = obj
		#resetting the value of w
		for j in range(cols):
			w[j] -= eta * del_w[j]
	eta = best_eta
	#print("Best Eta:",eta)
	# use best_eta to calculate w
	for j in range(cols):
		w[j] +=eta * del_w[j]
	error = 0.0
	for t in range(rows):
		if (trainlabels.get(t) != None):
			temp = trainlabels[t]-w_transpose_x(w,data[t])
			error += float(temp**2)
	if (abs(prev_error - error) <= stop_cond):
		cond = False
		#print("cond",cond)
	prev_error = error

#	print("Error",error)

print("ETA:", eta)
#####################################
### Normalization 
#####################################

#print("w= ")
norw = 0
for j in range(cols-1):
	norw += w[j]**2
#print(w[j])
norw = math.sqrt(norw)
#print("||w|| = ",norw)
dist_from_origin = w[len(w) - 1]/norw
#print("Distance to origin = ", dist_from_origin)

################################################
###  Prediction
################################################
for i in range(0,rows,1):
	if(trainlabels.get(i) == None):
		y = w_transpose_x(w,data[i])

		#print(y)
		if(y>0):
			print("1 ",i)
		else:
			print("0 ",i)
