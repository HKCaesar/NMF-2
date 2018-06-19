import numpy as np

def makeRandom(m,n):
    return np.random.rand(m,n)

def readTSV(name):
	t = []
	f = open(name).read().split("\n")[1:-1]
	for i in range(len(f)):
		t.append(list(map(int,f[i].split("\t")[1:])))
	return np.array(t)

print(readTSV("mutation-counts.tsv"))
