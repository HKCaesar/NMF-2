import util
import multiplicative_update as mu
import numpy as np
import plotting
import matplotlib.pyplot as plt
import random

V = util.readTSV("data/classdata/mutation-counts.tsv")

actualW = np.load("data/classdata/example-signatures.npy")

k = 5
W,H = mu.multiplicativeUpdate(V,k)

for i in range(1):
    WP,HP = mu.nonSmooth(V,k,random.random())
    if(mu.f(V,WP,HP)<mu.f(V,W,H)):
        W,H = WP, HP

score = plotting.cosineTable(H,actualW,showBest=True)
plt.show()

