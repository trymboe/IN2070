from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import *
from a0 import standardiser_kontrast

f = imread('portrett.png', as_gray=True)
maske = imread('geometrimaske.png', as_gray=True)
N, M = f.shape
N2, M2 = maske.shape

"""
dx, dy = 0, 0
translate = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

theta = radians(-15)
rotate = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

cx, cy = N2/N, M2/M
scale = np.array([[cx, 0, 0], [0, cy, 0], [0, 0, 1]])

rot_scale = rotate.dot(scale)
T = rot_scale.dot(translate)
"""
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
print(T)

def NN_interpolation(pixel):
    pass

def BL_interpolation(pixel):
    pass

def forwards_mapping(f, T):
    f_out = np.zeros((N2,M2))
    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = T.dot(vec_in)
            x = int(vec_out[0])
            y = int(vec_out[1])
            if x in range(N2) and y in range(M2):
                f_out[x, y] = f[i, j]
    return f_out

def backwards_mapping_NM(f_in, transform):

    f_out = np.zeros((N2,M2))

    for i in range(N2):
        for j in range(M2):
            vec_in = np.array([i, j, 1])
            vec_out = np.dot(transform, vec_in)
            x = int(vec_out[0])
            y = int(vec_out[1])
            if (x in range(N) and y in range(M)):
                f_out[i, j] = f_in[x, y]
    return f_out

# Viser de forskjellige bildene med standardisert kontrast fra oppgave 1.1
plt.figure()
f = standardiser_kontrast(f)
f_forlengs = forwards_mapping(f, T)
f_baklengs = backwards_mapping_NM(f, T)
plt.imshow(maske)
plt.imshow(f_forlengs, cmap="gray", alpha = 1)
plt.title('Forlengs mapping')
plt.figure()
plt.imshow(f_baklengs, cmap="gray", alpha = 1)
plt.title('baklengs mapping')

plt.show()
