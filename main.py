import util
import multiplicative_update as mu
import numpy as np
import plotting
import matplotlib.pyplot as plt
import random
import automatic_relevance as ard
from ardnmf import ARDNMF
model = ARDNMF(a=0)

V = util.readTSV("data/alexsandrov/breast.txt").T

with open("data/alexsandrov/signatures.txt") as f:
    actualW = f.read().split("\n")[1:]
    for i in range(len(actualW)):
        actualW[i] = list(map(float,actualW[i].split("\t")[3:]))
    actualW = np.array(actualW).T

k = 27
H = model.fit_transform(V)
W = model.components_


W,H = ard.ard(V,k,0,1,10**(-5),1,max_iter=300)
H = util.addZeros(H,actualW.shape[0]-H.shape[0])
actualW = util.addZeros(actualW,H.shape[0]-actualW.shape[0])

score = plotting.cosineTable(H,actualW,showBest=True)
print(score)
plt.show()
