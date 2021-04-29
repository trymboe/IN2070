from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import *
from task2a import konvolusjon

def gauss_filter(sigma=1):
    N = M = ceil(8*int(sigma)+1) # Om vi ikke runder av sigma kan vi få partall på sidelengden
    h = np.zeros((N, M))
    # regne ut A
    for x in range(N):
        for y in range(M):
            h[x,y] = np.exp(-(x**2 + y**2)/(2*sigma**2))
    A = 1/sum(h.flatten())
    for x in range(N):
        for y in range(M):
            h[x,y] = A*h[x,y]
    return h

def calc_gradient(f):
    N, M = f.shape
    gm = np.zeros((N, M)) # Gradient magnitude
    gd = np.zeros((N, M)) # Gradient direction
    for i in range(N-1):
        for j in range(M-1):
            gx = f[i+1][j] - f[i-1][j]
            gy = f[i][j+1] - f[i][j-1]
            gm[i][j] = sqrt(gx**2 + gy**2)
            gd[i][j] = atan2(gy, gx)
    return gm, gd

def trim_edges(gm, gd):
    N, M = gm.shape
    n1, n2, = 0, 0
    for i in range(1, N-1):
        for j in range(1, M-1):
            if gd[i, j] <= np.pi/8: # langs x-aksen
                n1 = gm[i-1, j]
                n2 = gm[i+1, j]
            elif 3*np.pi/8 >= gd[i, j] > np.pi/8: # høyre øvre hjørne
                n1 = gm[i-1, j-1]
                n2 = gm[i+1, j+1]
            elif 5*np.pi/8 >= gd[i, j] > 3*np.pi/8: # langs y-aksen
                n1 = gm[i, j-1]
                n2 = gm[i, j+1]
            elif 7*np.pi/8 >= gd[i, j] > 5*np.pi/8: # ventstre øvre hjørne
                n1 = gm[i-1, j+1]
                n2 = gm[i+1, j-1]
            if gm[i, j] <= n1 or gm[i, j] <= n2:
                gm[i, j] = 0
    return gm

def check8(f, x, y):
    for i in range(-1, 1):
        for j in range(-1, 1):
            if f[x+i, y+j] == 255:
                return True
    return False

def hysterese(f, tl, th):
    N, M = f.shape
    f_out = np.zeros((N, M))
    diff = 1
    while diff != 0:
        diff = 0
        for i in range(N):
            for j in range(M):
                if f[i, j] >= th and f_out[i, j] == 0:
                    f_out[i, j] = 255
                    diff += 1
                elif th > f[i, j] >= tl and check8(f_out, i, j) and f_out[i, j] == 0:
                    f_out[i, j] = 255
                    diff += 1
    return f_out

f = imread('cellekjerner.png', as_gray=True)
N, M = f.shape
# Symetriske matriser
"""
Endte opp med å ikke bruke disse, da konvolusjon tar veldig lang tid. Men kunne
eventuelt lage et gx bilde med å skrive gx = konvolusjon(f, hx), og tilsvarende
med gy, men følte det ble unødvendig bruk av tid.
"""
hx = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
hy = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
# Utfører cannys algoritme
filtrert = konvolusjon(f, gauss_filter(1.2))
print("Lavpassfilter ferdig")
gradient, direction = calc_gradient(filtrert)
print("Gradient ferdig")
thinned = trim_edges(gradient, direction)
print("Trimming av kanter ferdig")
final = hysterese(thinned, 2, 46)
print("Hysterese ferdig")
# Sammenligner resultatet med detekterte_kanter
plt.figure()
plt.imshow(final, cmap="gray")
plt.title('Hysterese')
plt.savefig("task2c.png")

plt.figure()
dk = imread('detekterte_kanter.png', as_gray=True)
plt.imshow(dk, cmap="gray")
plt.title("detekterte kanter")

plt.show()
