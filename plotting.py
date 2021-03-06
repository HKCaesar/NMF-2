import matplotlib.pyplot as plt
import numpy as np
import util
from scipy.optimize import linear_sum_assignment

def cosineTable(a,b,showBest=False):
    """Show Best runs the Hungarian algorithm to determine the best matching"""

    similarities = [[util.cosineSimilarity(a[i],b[j]) for j in range(len(b))] for i in range(len(a))]
    similarities = np.array(similarities)

    numSignatures = len(similarities)

    s = util.hungarian(a,b)

    if(showBest):
        row,col = linear_sum_assignment(-1*similarities)
        best = [[False for i in range(numSignatures)] for j in range(numSignatures)]

        for i in range(len(row)):
            best[row[i]][col[i]] = True
      
        for i in range(numSignatures):
            for j in range(numSignatures):
                if(not(best[i][j])):
                    similarities[i,j] = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = ax.matshow(similarities)
    fig.colorbar(m)
    
    plt.ylabel("My Signature")
    plt.xlabel("Actual signature")
    ax.set_xticklabels(['']+list(range(1,numSignatures+1)))
    ax.set_yticklabels(['']+list(range(1,numSignatures+1)))

    return s/numSignatures

def kasarBar(H):
    #Converting the order of the kasar graphs to the order stored in the table
    mutations = "	A[C>A]A	A[C>A]C	A[C>A]G	A[C>A]T	A[C>G]A	A[C>G]C	A[C>G]G	A[C>G]T	A[C>T]A	A[C>T]C	A[C>T]G	A[C>T]T	A[T>A]A	A[T>A]C	A[T>A]G	A[T>A]T	A[T>C]A	A[T>C]C	A[T>C]G	A[T>C]T	A[T>G]A	A[T>G]C	A[T>G]G	A[T>G]T	C[C>A]A	C[C>A]C	C[C>A]G	C[C>A]T	C[C>G]A	C[C>G]C	C[C>G]G	C[C>G]T	C[C>T]A	C[C>T]C	C[C>T]G	C[C>T]T	C[T>A]A	C[T>A]C	C[T>A]G	C[T>A]T	C[T>C]A	C[T>C]C	C[T>C]G	C[T>C]T	C[T>G]A	C[T>G]C	C[T>G]G	C[T>G]T	G[C>A]A	G[C>A]C	G[C>A]G	G[C>A]T	G[C>G]A	G[C>G]C	G[C>G]G	G[C>G]T	G[C>T]A	G[C>T]C	G[C>T]G	G[C>T]T	G[T>A]A	G[T>A]C	G[T>A]G	G[T>A]T	G[T>C]A	G[T>C]C	G[T>C]G	G[T>C]T	G[T>G]A	G[T>G]C	G[T>G]G	G[T>G]T	T[C>A]A	T[C>A]C	T[C>A]G	T[C>A]T	T[C>G]A	T[C>G]C	T[C>G]G	T[C>G]T	T[C>T]A	T[C>T]C	T[C>T]G	T[C>T]T	T[T>A]A	T[T>A]C	T[T>A]G	T[T>A]T	T[T>C]A	T[T>C]C	T[T>C]G	T[T>C]T	T[T>G]A	T[T>G]C	T[T>G]G	T[T>G]T"
    mutations = mutations.split("\t")[1:]

    kasarOrder = "T>C T>G T>A C>A C>G C>T".split(" ")
    lexOrder = []
    for i in 'ACTG':
        for j in 'ACTG':
            lexOrder.append(i+j)
    lexOrder = sorted(lexOrder)
    print(lexOrder)
    mutations = sorted(mutations,key=lambda x: kasarOrder.index(x.split("[")[1].split("]")[0]) + lexOrder.index(x[0]+x[-1])/100)

    numRows = len(H)*100+11

    for i in range(len(H)):
        l = list(H[i])
        yPos = np.arange(len(l))
        plt.subplot(numRows+i)
        barList = plt.bar(yPos,l,align='center',alpha=0.5)
        colors = ['b', 'g', 'y', 'c', 'm', 'r']
        for j in range(0,len(l)):
            barList[j].set_color(colors[j//16])
        
        
    plt.ylabel('Amount')
    plt.show()
