from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

f = imread('mona.png', as_gray=True)


def r_transformation(f_in, transform):
    N,M = f_in.shape
    f_out = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = np.dot(transform, vec_in)
            x = int(vec_out[0])
            y = int(vec_out[1])
            if (x in range(N) and y in range(M)):
                f_out[i, j] = f_in[x, y]
    return f_out

def lag_rot(f_in, transform):
    N,M = f_in.shape
    sentrumX   = round(((N+1)/2)-1)
    sentrumY   = round(((M+1)/2)-1)
    m_1 = np.array([[1, 0, sentrumX],
              [0, 1, sentrumY],
              [0, 0, 1]])
    m_2 = np.array([[1, 0, -sentrumX],
              [0, 1, -sentrumY],
              [0, 0, 1]])
    rot_matrise = m_1.dot(transform.dot(m_2))

    return rot_matrise

th = np.pi/9 # 10 grader

rotate = np.array([[np.cos(th), -np.sin(th), 1],
                   [np.sin(th), np.cos(th), 1],
                   [0, 0, 1]])

rotateInv = np.linalg.inv(rotate)

new = lag_rot(f, rotateInv)

g = r_transformation(f, new)
plt.figure()
plt.title("Original")
plt.imshow(f, cmap='gray')
plt.figure()
plt.title("Transformert:)")
plt.imshow(g, cmap='gray')
plt.show()
