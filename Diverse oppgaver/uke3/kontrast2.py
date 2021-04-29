from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import e

f = imread('mona.png', as_gray=True)

N,M = f.shape
#print(f[5,10])
f_out = np.zeros((N,M))
f_out1 = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        f_out[i,j] = f[i,j]-round(255/2)
        f_out[i,j] = (f_out[i,j]*1.7)+(255/2)
        f_out1[i,j] = f_out[i,j] + 50

fig1, (ax2, ax3, ax4) = plt.subplots(3)
ax2.imshow(f, cmap="gray")
ax3.imshow(f_out, cmap="gray", vmin=0, vmax=255)
ax4.imshow(f_out1, cmap="gray", vmin=0, vmax=255)
plt.show()
