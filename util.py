import numpy as np
from scipy.optimize import linear_sum_assignment

def makeRandom(m,n):
    return np.random.rand(m,n)

def readTSV(name):
	t = []
	f = open(name).read().split("\n")[1:-1]
	for i in range(len(f)):
		t.append(list(map(int,f[i].split("\t")[1:])))
	return np.array(t)

def cosineSimilarity(a,b):
    top = sum([a.item(i)*b.item(i) for i in range(a.size)])
    bottom = sum([a.item(i)**2 for i in range(a.size)])**.5
    bottom*=sum([b.item(i)**2 for i in range(b.size)])**.5
    return top/bottom

def hungarian(a,b):
    similarities = [[cosineSimilarity(a[i],b[j]) for j in range(len(b))] for i in range(len(a))]
    similarities = np.array(similarities)

    row,col = linear_sum_assignment(-1*similarities)
    s = sum([similarities[row[i]][col[i]] for i in range(len(row))])

    for i in range(len(row)):
        s+=similarities[row[i]][col[i]]

    return s/len(row)                

def addZeros(arr,num):
    #Add zero rows to end of array
    for i in range(num):
        arr = np.append(arr,np.array([0 for i in range(arr.shape[1])]),axis=0)
    return arr
