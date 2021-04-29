from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

f = imread('mona.png',as_gray=True)
plt.imshow(f, cmap="gray")

N,M = f.shape
deg = 45
rad = math.radians(deg)
cos = math.cos(rad)
sin = math.sin(rad)
f_out = np.zeros((N,M))

sentrumX   = round(((N+1)/2)-1)
sentrumY   = round(((M+1)/2)-1)

roterpunktX = sentrumX
roterpunktY = sentrumY

for i in range(N):
    for j in range(M):
        x=N-1-i-roterpunktX
        y=M-1-j-roterpunktY

        ny_x=round(-y)
        ny_y=round(x)

        ny_x=sentrumX-ny_x
        ny_y=sentrumY-ny_y

        if(ny_x < N and ny_x >0 and ny_y > 0 and ny_y < M):

            f_out[ny_y][ny_x] = f[i][j]

plt.figure()
plt.imshow(f_out, cmap = "gray")
plt.show()
