from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import *
from task1a import standardiser_kontrast

f = imread('portrett.png', as_gray=True)
maske = imread('geometrimaske.png', as_gray=True)
N, M = f.shape
N2, M2 = maske.shape

# Fant verdiene ved å se på matplotlib og se koordinatene
# Høyre øye, venstre øye, høyre munnvik, venstre munnvik
x1, x2, x3, x4 = 118, 84, 142, 115
y1, y2, y3, y4 =  67, 88, 100, 116
x1_t, x2_t, x3_t, x4_t = 342, 166, 317, 192
y1_t, y2_t, y3_t, y4_t =  258, 258, 440, 440
# Setter sammen G og d for å finne a og b til rotasjonsmatrisen
#G = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1], [x4, y4, 1]])
G = np.array([[y1, x1, 1], [y2, x2, 1], [y3, x3, 1], [y4, x4, 1]])
d1 = np.array([[x1_t], [x2_t], [x3_t], [x4_t]])
d2 = np.array([[y1_t], [y2_t], [y3_t], [y4_t]])
m = G.T.dot(G)
m = np.linalg.inv(m).dot(G.T)
# Sette sammen rotasjonsmatrisen:
a = m.dot(d1)
b = m.dot(d2)
T = np.array([np.ravel(b), np.ravel(a), [0, 0, 1]])

def calc_bl(f, x, y):
    x0 = floor(x)
    y0 = floor(y)
    x1 = ceil(x)
    y1 = ceil(x)
    dx = x - x0
    dy = y - y0
    p = f[x0][y0] + (f[x1][y0] - f[x0][y0])*dx
    q = f[x0, y1] + (f[x1, y1] - f[x0, y1])*dx
    return p + (q - p)*dy

def forwards_mapping(f, T):
    f_out = np.zeros((N2,M2))
    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = T.dot(vec_in)
            x = round(vec_out[0])
            y = round(vec_out[1])
            if x in range(N2) and y in range(M2):
                f_out[x, y] = f[i, j]
    return f_out

def BM_NN(f, T):
    T_I = np.linalg.inv(T)
    f_out = np.zeros((N2,M2))
    for i in range(N2):
        for j in range(M2):
            vec_out = np.array([i, j, 1])
            vec_in = np.dot(T_I, vec_out)
            x = round(vec_in[0])
            y = round(vec_in[1])
            if (x in range(N) and y in range(M)):
                f_out[i, j] = f[x, y]
    return f_out

def BM_BL(f, T):
    T_I = np.linalg.inv(T)
    f_out = np.zeros((N2,M2))
    for i in range(N2):
        for j in range(M2):
            vec_out = np.array([i, j, 1])
            vec_in = np.dot(T_I, vec_out)
            x = round(vec_in[0])
            y = round(vec_in[1])
            if (x in range(N) and y in range(M)):
                f_out[i, j] = calc_bl(f, x, y)
    return f_out

# Viser de forskjellige bildene med standardisert kontrast fra oppgave 1.1
f = standardiser_kontrast(f)

plt.figure()
f_forlengs = forwards_mapping(f, T)
plt.imshow(maske)
plt.imshow(f_forlengs, cmap="gray", alpha = 1)
plt.title('Forlengs mapping')
plt.savefig("forlengs.png")

plt.figure()
f_NN = BM_NN(f, T)
plt.imshow(maske)
plt.imshow(f_NN, cmap="gray", alpha = 1)
plt.title('Baklengs mapping med NN interpolasjon')
plt.savefig("baklengs_NN.png")

plt.figure()
f_BL = BM_BL(f, T)
plt.imshow(maske)
plt.imshow(f_BL, cmap="gray", alpha = 1)
plt.title('Baklengs mapping med bilineær interpolasjon')
plt.savefig("baklengs_BL.png")

plt.show()
