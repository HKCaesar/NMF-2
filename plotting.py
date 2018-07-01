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
