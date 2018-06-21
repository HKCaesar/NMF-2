import util
import multiplicative_update as mu
import numpy as np
import plotting
import matplotlib.pyplot as plt
import random

V = util.readTSV("data/alexsandrov/breast.txt")

with open("data/alexsandrov/signatures.txt") as f:
    actualW = f.read().split("\n")[1:]
    for i in range(len(actualW)):
        actualW[i] = list(map(float,actualW[i].split("\t")[3:]))
    actualW = np.array(actualW)

W,H = mu.multiplicativeUpdate(V,27)

for i in range(1):
    print(i)
    WP,HP = mu.nonSmooth(V,27,random.random())
    if(mu.f(V,WP,HP)<mu.f(V,W,H)):
        W,H = WP, HP

score = plotting.cosineTable(W.transpose(),actualW.transpose(),showBest=True)
print(score)
plt.show()

