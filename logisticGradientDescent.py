##################################################
###	Gradient Descent Algorithm
###     - Logistic discrimination  
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
		y += w[j]*data[j]
	#print(y)
	return(y)

eta = 0.001       # eta for climate 
stop_cond = 0.001  # stp_cond for climate 
#eta = 0.01       # eta for toy dataset
#stop_cond = 0.0000001 # stop_cond for toy dataset 

prev_error = float('inf') 
#print("prev_error",prev_error)
cond = True
while (cond == True):
	del_w = []
	for j in range(0,cols,1):
		del_w.append(0)
	for t in range(0,rows,1):
		y = 0.0
		if(trainlabels.get(t) != None):
			y = w_transpose_x(w,data[t])
			for i in range(cols):
				del_w[i] += (trainlabels[t] - (1.0/(1 + math.exp(-y))))* data[t][i]
	for j in range(cols):
		w[j] += eta * del_w[j]
#	print(w)	
	error = 0.0
	for t in range(rows):
		if(trainlabels.get(t) != None):
			y = w_transpose_x(w,data[t])
			if (trainlabels[t] == 0):  
				error += -1 * math.log(1 - (1.0/(1 + math.exp(-y))))  
			else :	
				error += -1 * math.log(1.0/(1 + math.exp(-y)))
				
	if(abs(prev_error - error) <= stop_cond):
		cond = False
	prev_error = error
	#print("Error",error)

#####################################
### Normalization 
#####################################

#print("w= ")
norw = 0
for j in range(cols-1):
	norw += w[j]**2
#	print(w[j])
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
			print("1",i)
		else:
			print("0",i)
