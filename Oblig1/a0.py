from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

def lagHistogram(f):
    array = [0] * G
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

def finnMiddelverdi(f):
    return np.median(f)

def finnStd(f):
    return np.sqrt(np.var(f))


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


def standardiser_kontrast(f):
    h = lagHistogram(f)
    n = normalisertHist(h)
    middelverdi = finnMiddelverdi(f)
    std = finnStd(f)
    print(middelverdi, std)
    #ønsket middelverdi og standardavvik
    middelverdi_t = 127
    std_t = 64
    #transformasjonen
    t = transformasjon(middelverdi, std, middelverdi_t, std_t)

    #Bilde etter transformasjon
    f_ny = utforTransformasjon(f,t)

    h_ny = lagHistogram(f_ny)
    n_ny = normalisertHist(h_ny)
    middelverdi_ny = finnMiddelverdi(f_ny)
    std_ny = finnStd(f_ny)
    return f_ny


#print(middelverdi_ny, std_ny)
"""y_pos = np.arange(G)


fig1, (ax1, ax2) = plt.subplots(2)
ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
ax2.bar(y_pos, h)

fig2, (ax1, ax2) = plt.subplots(2)
ax1.imshow(f_ny, cmap="gray", vmin=0, vmax=255)
ax2.bar(y_pos, h_ny)

plt.show()
"""
