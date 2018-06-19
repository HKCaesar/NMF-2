import numpy as np

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
