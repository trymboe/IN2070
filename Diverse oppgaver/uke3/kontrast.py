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
    out = [y_pos, array]
    return out

N,M = f.shape
#print(f[5,10])
f_out = np.zeros((N,M))

a, b = 1.5, 20
for i in range(N):
    for j in range(M):
        if f[i][j] * a + b > 255:
            f_out[i][j] = 255
        else:
            f_out[i][j] = f[i][j] * a + b

out1 = lagHistogram(f)
out2 = lagHistogram(f_out)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(out1[0], out1[1])
ax2.imshow(f, cmap="gray")

#np.clip(f_out, 0, 255)

fig1, (ax3, ax4) = plt.subplots(1,2)
ax3.bar(out2[0], out2[1])
ax4.imshow(f_out, cmap="gray", vmin=0, vmax=255)
"""plt.imshow(f, cmap="gray")
plt.figure()
plt.imshow(f_out, cmap="gray", vmin=0, vmax=255)"""
plt.show()
