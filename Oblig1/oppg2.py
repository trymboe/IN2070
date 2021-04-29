from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

f = imread('cellekjerner.png', as_gray=True)


filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def addPadding(f):
    N,M = f.shape
    f_exp = np.zeros((N+2,M+2))

    #legger inn-bildet inn i det større bilde
    for i in range(N):
        for j in range(M):
            f_exp[i+1, j+1] = f[i,j]
    #finner nærmeste nabo
    for x in range(4):
        for i in range(1, N+1):
            f_exp[i,0] = f[i-1,0]
            f_exp[i,M+1] = f[i-1, M-1]
        for i in range(1,M+1):
            f_exp[0,i] = f[0, i-1]
            f_exp[N+1,i] = f[N-1, i-1]
    #legger inn hjørnene
    f_exp[0,0] = f[0,0]
    f_exp[0,M+1] = f[0,M-1]
    f_exp[N+1,0] = f[N-1,0]
    f_exp[N+1,M+1] = f[N-1,M-1]

    return f_exp

def applyConvolution(f, filter):
    N,M = f.shape
    X,Y = filter.shape
    f_out = np.zeros((N,M))
    for i in range(1,N-1):
        for j in range(1,M-1):
            sum = 0
            for x in range(-1,2):
                for y in range(-1,2):
                    sum += f[i+x, j+y]
            f_out[i,j] = sum/(X*Y)

    return f_out




f_exp = addPadding(f)
f_out = applyConvolution(f_exp, filter)


plt.imshow(f_exp, cmap="gray", vmin=0, vmax=255)
plt.figure()
plt.imshow(f_out, cmap="gray", vmin=0, vmax=255)
plt.show()
