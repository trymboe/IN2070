from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import *

def lagHistogram(f):
    G = int(max(map(max, f)))+2
    array = [0] * G
    for i in f:
        for p in i:
            array[round(p)] += 1
    return array, G

def middelverdi(h):
    tot = 0
    for i in range(len(h)):
        tot += i*h[i]
    return round(tot/sum(h))

def finn_std(h):
    std = 0
    mv = middelverdi(h) #middelverdi
    for i in range(len(h)):
        std += (i-mv)**2*h[i]
    return round(sqrt(std/sum(h)))

def GTT(mu, std, mu_t, std_t, G=256):
    T = np.zeros(G)
    for i in range(G):
        T[i] = mu_t + (i-mu)*std_t/std
    return T

def utbilde(f, T):
    N, M = f.shape
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_out[i][j] = T[int(f[i][j])]
    return f_out

def standardiser_kontrast(f):
    f = imread('portrett.png', as_gray=True)
    h, G = lagHistogram(f)
    mu = middelverdi(h)
    std = finn_std(h)
    T = GTT(mu, std, 127, 64, G)
    f_out = utbilde(f, T)
    return f_out

if __name__ == "__main__":
    # Lage bilde og sette N, M og G
    f = imread('portrett.png', as_gray=True)
    N,M = f.shape
    # Lager histogram med tilhørende middelverdi og standardavvik
    h, G = lagHistogram(f)
    mu = middelverdi(h)
    std = finn_std(h)
    print("Gamle verdier")
    print(mu, std)
    T = GTT(mu, std, 127, 64, G)
    # Kutte verdier utenfor [0, 255]
    # Lage utbilde
    f_out = utbilde(f, T)
    # Lage nytt histogram med tilhørende middelverdi og standardavvik
    h_out, G2 = lagHistogram(f_out)
    ny_mu = middelverdi(h_out)
    ny_std = finn_std(h_out)
    print("Nye verdier")
    print(ny_mu, ny_std)
    # Viser bildene og de tilhørende histogrammene
    fig1, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
    ax2.bar(np.arange(G), h)

    fig2, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(f_out, cmap="gray", vmin=0, vmax=255)
    ax2.bar(np.arange(G2), h_out)

    plt.show()
    plt.imshow(f_out, cmap="gray")
    plt.savefig("standardiser_kontrast.png")
