from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import e

f = imread('mona.png', as_gray=True)
N,M = f.shape
G=256
f_out = np.zeros((N,M))

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

    for i in range(G):
        t[i] = (G-1)*c[i]
    return t

def normalisertHist(hist):
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p

h = lagHistogram(f)
c = lagKumulativt(h)
c_normal = c*max(h)/max(c)
n = normalisertHist(c)
t = transformer(n)

for i in range(N):
    for j in range(M):
        f_out[i,j] = t[int(f[i,j])]




"""nyHist = np.zeros(G)
for i in range(G):
    nyHist[i] = t[int(c_normal[i])]"""


y_posH = np.arange(len(h))
y_posC = np.arange(len(c))
y_posT = np.arange(len(t))
y_posN = np.arange(len(n))
#y_posNyHist = np.arange(len(nyHist))



plt.bar(y_posH, h)
plt.plot(y_posC, c_normal, color = "red")
#plt.bar(y_posN, n)
plt.xlim([0,G-1])
plt.show()

fig1, (ax1, ax2, ax3) = plt.subplots(1,2)
ax1.imshow(f, cmap="gray")
ax2.imshow(f_out, cmap="gray")
ax3.plot(y_posC, c_normal, color = "red")
#ax3.plot(y_posNyHist, nyHist, color = "blue")
plt.show()
