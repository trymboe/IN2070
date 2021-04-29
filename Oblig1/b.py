from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

f = imread('portrett.png', as_gray=True)
maske = imread('geometrimaske.png')
N,M = f.shape
f_out = np.zeros((600, 512))

for i in range(N):
    for j in range(M):
        f_out[i,j] = f[i,j]


def transformation(f_in, transform):
    N,M = f_in.shape
    f_out = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = np.dot(transform, vec_in)
            x = int(vec_out[0])
            y = int(vec_out[1])
            if (x in range(N) and y in range(M)):
                f_out[x, y] = f_in[i, j]
    return f_out





translate = np.array([[3.4, 0, -400],
                      [0, 3.4, -310],
                      [0, 0, 1]])

th = np.pi/18 # 10 grader

rotate = np.array([[np.cos(-math.pi/6), -np.sin(-math.pi/6), 260],
                   [np.sin(-math.pi/6), np.cos(-math.pi/6), 170],
                   [0, 0, 1]])

scale = np.array([[0.5, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 1]])

rot_trans = rotate.dot(translate)
#f_out = transformation(f_out, translate)
f_out = transformation(f_out, rot_trans)

plt.imshow(maske)
plt.imshow(f_out, cmap="gray", alpha = 0.8)
plt.show()
