import numpy as np
from util import makeRandom
from copy import deepcopy

def e(beta):
    if(beta<=2):
        return 1/(3-beta)
    return 1/(beta-1)

def d(beta,x,y):
    if(beta==1):
        return x * np.log(x/y)/np.log(10) - x + y
    if(beta==0):
        return x/y - np.log(x/y)/np.log(10)-1
    return x**beta/(beta*(beta-1)) + y**beta/beta - (x*y**(beta-1))/(beta-1)

def D(beta,V,WH):
    return sum([d(beta,V.item(i),WH.item(i)) for i in range(V.size)])

def p(k,n,W,V,VH,beta):
    r = 0
    for f in range(W.shape[0]):
        r+=W[f,k]*V[f,n]*VH[f,n]**(beta-2)
    return r

def q(k,n,W,H,V,VH,beta):
    r = 0
    for f in range(W.shape[0]):
        r+=W[f,k] * VH[f,n]**(beta-1)
    return r

def repmat(lamb,a,n):
    mat = 0
    if(a==1):
        mat = makeRandom(lamb.size,n)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i,j] = lamb[i]
    else:
        mat = makeRandom(a,lamb.size)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i,j] = lamb[j]

    return mat

def ard(V,k,a,beta,tau,phi):
    f = V.shape[0]
    n = V.shape[1]

    W = makeRandom(f,k)
    H = makeRandom(k,n)
    lamb = makeRandom(k,1)

    tol = -100000
    previousTol = 10

    averageV = 1/(f*n) * sum([V.item(i) for i in range(f*n)])

    b = np.pi * (a-1)*averageV/(2*k)
    c = (f+n)/2 + a+1

    for i in range(k):
        pr = ((1/2 * sum([W[fi,i]**2 for fi in range(f)])
                     + 1/2 * sum([H[i,ni]**2 for ni in range(n)]))+b)/c
        lamb[i,0] = pr


    while(abs(tol-previousTol)>tau):
        previousTol = tol
        top = W.transpose().dot(
            ((W.dot(H))**(beta-2))*V)

        bottom = W.transpose().dot(W.dot(H)**(beta-1))+phi * H/repmat(lamb,1,n)

        HC = deepcopy(H)
        for i in range(k):
            for j in range(n):
                HC[i,j] = q(i,j,W,H,V,W.dot(H),beta) + phi/lamb[i] * HC[i,j]
            
        HP = H*(top/bottom)**(e(beta))

        top = ((W.dot(H))**(beta-2)*V).dot(H.transpose())
        bottom = (W.dot(H)**(beta-1)).dot(H.transpose())+phi * W/repmat(lamb,f,1)
        W = W*(top/bottom)**(e(beta))
        H = HP

        tol = -10000

        for i in range(k):
            r = 0
            for j in range(f):
                r+=1/2 * W[j,i]**2
            for j in range(n):
                r+=1/2 * H[i,j]**2
            r+=b
            r/=c

            tol = max(r,abs((r-lamb[i,0])/lamb[i,0]))
            lamb[i,0] = r

    return (W,H)

        
