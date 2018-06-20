import util
import multiplicative_update as mu
import numpy as np

V = util.readTSV("data/mutation-counts.tsv")
W,H = mu.multiplicativeUpdate(V,5)

actualH = np.load("data/example-signatures.npy")

differences = [[util.cosineSimilarity(H[i],actualH[j]) for j in range(len(actualH))] for i in range(len(H))]
differences = np.array(differences)

print(differences)
        
