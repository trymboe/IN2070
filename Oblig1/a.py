from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

def lagHistogram(f):
    maks = 256
    for i in range(N):
        for j in range(M):
            if f[i,j] > maks:
                maks = int(f[i,j])
    array = [0] * maks
    for i in range(N):
        for j in range(M):
            value = round(f[i,j])
            if(value in range(0, maks)):
                array[value] = array[value]+1
    y_pos = np.arange(len(array))
    return array

def normalisertHist(hist):
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p


def finnMiddelverdi(normHist):
    middelverdi = 0
    for i in range(len(normHist)):
        middelverdi += normHist[i]*i
    return middelverdi

def finnStd(normHist,f):
    v_1 = 0
    v_2 = finnMiddelverdi(normHist)**2
    for i in range(len(normHist)):
        v_1 += i**2*normHist[i]
    varians = v_1 - v_2
    return math.sqrt(varians)




def transformasjon(m, s, m_t, s_t):
    t = np.zeros(G)
    
    for i in range(G):
        t[i] = m_t+(i-m)*(s_t/s)
    return t

def utforTransformasjon(f, t):
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_out[i,j] = t[int(f[i,j])]
    return f_out

f = imread('portrett.png', as_gray=True)
N,M = f.shape
G = 256




h = lagHistogram(f)
n = normalisertHist(h)
middelverdi = finnMiddelverdi(n)
std = finnStd(n,f)
print("values for input image: Mean", middelverdi, "Standard diviation", std)
#ønsket middelverdi og standardavvik
middelverdi_t = 127
std_t = 64
#transformasjonen
t = transformasjon(middelverdi, std, middelverdi_t, std_t)

#Bilde etter transformasjon
f_ny = utforTransformasjon(f,t)

h_ny = lagHistogram(f_ny)
n_ny = normalisertHist(h_ny)
middelverdi_ny = finnMiddelverdi(n_ny)
std_ny = finnStd(n_ny,f_ny)


print("values for output image: Mean", middelverdi_ny, "Standard diviation", std_ny)
y_pos = np.arange(G)
y_pos2 = np.arange(len(n_ny))


fig1, (ax1, ax2) = plt.subplots(2)
ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
ax2.bar(y_pos, h)


fig2, (ax1, ax2) = plt.subplots(2)
ax1.imshow(f_ny, cmap="gray", vmin=0, vmax=255)
ax2.bar(y_pos2, h_ny)


plt.show()
