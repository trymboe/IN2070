from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import e

f = imread('mona.png', as_gray=True)
N,M = f.shape
G=256

def lagHistogram(f):
    array = [0] * G
    for i in range(N):
        for j in range(M):
            value = round(f[i,j])
            if(value in range(0, G-1)):
                array[value] = array[value]+1
    y_pos = np.arange(len(array))
    return array

def lagKumulativt(hist):
    c = np.zeros(G)
    c[0] = hist[0]
    for i in range(1, len(hist)):
        c[i] = c[i-1] + hist[i]
    return c

def transformer(c):
    t = np.zeros(G)
    for i in range(G-1):
        t[i] = (G-1)*c[i]
    return t

def normalisertHist(hist):
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p

h = lagHistogram(f)
c_normal = c*max(h)/max(c)
n = normalisertHist(h)
t = transformer(n)
c = lagKumulativt(n)
y_posH = np.arange(len(h))
y_posC = np.arange(len(c))
y_posT = np.arange(len(t))
y_posN = np.arange(len(n))



#plt.bar(y_posH, h)
plt.bar(y_posN, n)
#plt.bar(y_posT, t)
plt.xlim([0,G-1])
plt.show()

"""fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.bar(y_posH, h)
ax2.bar(y_posC, c)
ax3.bar(y_posT, t)
plt.show()"""
