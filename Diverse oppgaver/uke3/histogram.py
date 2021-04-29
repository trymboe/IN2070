"""from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

f = imread('mona.png', as_gray=True)

#f.histogram()
N,M = f.shape
f_out = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        f_out[i,j] = f[i,j] - f[i-1,j]

plt.hist(f_out)
plt.title("histogram")
#plt.imshow(f_out, cmap="gray")
#plt.figure()
#plt.imshow(f, cmap="gray")
plt.show()
"""

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


#lagHistogram(f)

N,M = f.shape
#print(f[5,10])
f_out = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        #break
        f_out[i,j] = f[i,j]-round(255/2)
        f_out[i,j] = (f_out[i,j]*1.7)+(255/2)

#lagHistogram(f_out)
plt.imshow(f, cmap="gray")
plt.figure()
plt.imshow(f_out, cmap="gray", vmin=0, vmax=255)
plt.show()
