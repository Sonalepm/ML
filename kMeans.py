#################################################
###         K - means clustering
#################################################

import sys
import random
import math

datafile = sys.argv[1]
k = int(sys.argv[2])

f = open(datafile)
l = f.readline()
data = []

################################################
### Read datafile
################################################

while (l != ''):
	a = l.split()
	l2 = []
	for i in range(len(a)):
		l2.append(float(a[i]))
	data.append(l2)
	l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()
#print("rows",rows)
#print("cols",cols)
#print("K",k)

m = [[0]*cols for x in range(k)]
#print(m)

rand = 0
for cluster in range(k):
	rand = random.randint(0,rows-1)
	#print(rand)
	m[cluster] = data[rand]
#print("Initial m",m)

prev =  [[0]*cols for x in range(k)]
#print(prev)

md = []
d = []
n = []

md = [0 for c in range(k)]
d = [0.1 for c in range(k)]
n = [0.1 for c in range(k)]

#print("md",md,"n",n,"d",d)

tot_dist = 1
clusters = []
label = {}
delta = 0

while (tot_dist > delta):
	for i in range(rows):
		d = []
		min_dist = 0	
		for j in range(k):
			d.append(0)
		for j in range(k):
			for c in range(cols):
				d[j] += math.pow((data[i][c] - m[j][c]),2)
		for j in range(k):
			d[j] = math.sqrt(d[j])
		#print("d",d)
		min_dist = min(d)
		#print(min_dist)
		for j in range(k):
			if (d[j] == min_dist):
				label[i] = j
				n[j] += 1
				break
	m = [[0]*cols for x in range(k)]
	for i in range(rows):
		for j in range(k):
			if(label.get(i) == j):
				for c in range(cols):
					t1 = m[j][c]
					t2 = data[i][c]
					m[j][c] = t1+t2

	for i in range(cols):
		for j in range(k):
			m[j][i] = m[j][i]/n[j]
	
	clusters = [int(x) for x in n]
	#print(clusters)
	n = [0.1]*k
	
	md = []
	for c in range(k):
		md.append(0)
	for c in range(k):
		for j in range(cols):
			md[c] += float(math.pow((prev[c][j] - m[c][j]),2))
		md[c] = math.sqrt(md[c])
	
	#print("md",md)
	prev = m
	tot_dist = 0
	for i in range(len(md)):
		tot_dist += md[i]
	#print("Total distance = ",tot_dist)

######################################
### Classify points
######################################

for i in range(rows):
	print(label[i],i)
