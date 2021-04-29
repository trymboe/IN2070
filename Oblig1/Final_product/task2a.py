from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import *

def rotate180(h):
    for x in h:
        x = x[::-1]
    h = np.flip(h)
    return h

def utvid_bilde(f, a, b):
    N, M = f.shape
    f_out = np.zeros((N+a, M+b))
    i, j = 0, 0
    for x in range(N+a):
        for y in range(M+b):
            if x < a: i = a
            elif x >= N: i = N-1
            else: i = x
            if y < b: j = b
            elif y >= M: j = M-1
            else: j = y
            f_out[x][y] = f[i][j]
    return f_out

def konvolusjon(f, h):
    h = rotate180(h)
    n, m = h.shape
    a, b = int((n-1)/2), int((m-1)/2)
    f = utvid_bilde(f, a, b)
    N, M = f.shape
    f_out = np.zeros((N-a*2,M-b*2))
    for x in range(a, N-a):
        for y in range(b, M-b):
            respons = 0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                        respons += h[s][t]*f[x+s][y+t]
            f_out[x-a][y-b] = respons
    return f_out

if __name__ == "__main__":
    # Definerer filter og tilhørende verdier
    a, b = 3, 3
    n, m = 2*a+1, 2*b+1
    filter = np.array([[1/(m*n)]*n]*m)
    # Definerer innbildet og tilhørende verdier
    f = imread('cellekjerner.png', as_gray=True)

    plt.figure()
    plt.imshow(f, cmap="gray")
    plt.title('Original')

    plt.figure()
    f_out = konvolusjon(f, filter)
    plt.imshow(f_out, cmap="gray")
    plt.title('Konvolusjon')


    plt.show()
