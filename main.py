import util
import multiplicative_update as mu
import numpy as np
import plotting
import matplotlib.pyplot as plt
import random
import automatic_relevance as ard
from ardnmf import ARDNMF
model = ARDNMF(a=0)

"""V = util.readTSV("data/alexsandrov/breast.txt").T

with open("data/alexsandrov/signatures.txt") as f:
    actualW = f.read().split("\n")[1:]
    for i in range(len(actualW)):
        actualW[i] = list(map(float,actualW[i].split("\t")[3:]))
    actualW = np.array(actualW).T"""

beta = 1

V = util.readTSV("data/kasar/combined.tsv")
V = np.add(V,.000001)


k=3
W,H = ard.ard(V,k,0,beta,10**(-5),1,max_iter=300,ltype="l1")

for l in range(10):
    WP,HP = ard.ard(V,k,0,beta,10**(-5),1,max_iter=300)
    if(ard.D(beta,V,W.dot(H))>ard.D(beta,V,WP.dot(HP))):
        W,H = WP, HP
    
plotting.kasarBar(H)
