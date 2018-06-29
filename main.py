import util
import multiplicative_update as mu
import numpy as np
import plotting
import matplotlib.pyplot as plt
import random
import automatic_relevance as ard

V = util.readTSV("data/classdata/mutation-counts.tsv")

actualW = np.load("data/classdata/example-signatures.npy")

k = 5
W,H = ard.ard(V,k,1,2,.001,1)

for i in range(1):
    WP,HP = ard.ard(V,k,1,2,.001,1)
    if(mu.f(V,WP,HP)<mu.f(V,W,H)):
        W,H = WP, HP

score = plotting.cosineTable(H,actualW,showBest=True)
plt.show()

