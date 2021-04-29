from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import e

f = imread('mona.png', as_gray=True)

def lagHistogram(f):
    N,M = f.shape
    array = [0] * 255
    for i in range(N):
        for j in range(M):
            value = round(f[i,j])
            if(value in range(0, 255)):
                array[value] = array[value]+1
    y_pos = np.arange(len(array))

    plt.figure()
    plt.bar(y_pos, array)

def transformer(f):
    N,M = f.shape
    f_out = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_out[i,j] = f[i,j]*1.2
    return f_out

#histogram = lagHistogram(f)


f_out = transformer(f)

plt.imshow(f, cmap="gray")
plt.figure()
plt.imshow(f_out, cmap="gray", vmin=0, vmax=255)
plt.show()
