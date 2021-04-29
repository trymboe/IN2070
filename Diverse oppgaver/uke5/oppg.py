from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy import diff
from math import e

f = imread('mona.png', as_gray=True)
N,M = f.shape
G=256
f_out = np.zeros((N,M))

def lagHistogram(f):
    array = [1] * G
    for i in range(N):
        for j in range(M):
            value = round(f[i,j])
            if(value in range(0, G-1)):
                array[value] = array[value]+1
    y_pos = np.arange(len(array))
    return array


def normalisertHist(hist):
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p

def lagKumulativt(hist):
    c = np.zeros(G)
    c[0] = hist[0]
    for i in range(1, len(hist)):
        c[i] = c[i-1] + hist[i]
    return c

h = normalisertHist(lagHistogram(f))
c = lagKumulativt(h)
y_posH = np.arange(len(h))


dydx = np.gradient(h)
for i in range(G-1):
    if (dydx[i]) == 0:
        print(i)

print("test")
terskel = 25


for i in range(N):
    for j in range(M):
        if h[int(f[i,j])] > 0.002:
            f_out[i,j] = 1
        else:
            f_out[i,j] = 0
    """    sum = 0
        for k in range(int(terskel/-2), int(terskel/2)):
           if int(f[i,j]+k) in range(0, G):
               sum += h[int(f[i,j]+k)]
    if sum/terskel > 0.03:
        f_out[i,j] = 1
    else:
        f_out[i,j] = 0"""
"""
plt.imshow(f_out, cmap="gray")
plt.figure()
plt.imshow(f, cmap="gray")
plt.show()
"""
