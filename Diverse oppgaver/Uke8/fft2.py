from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

f = imread('car.png', as_gray=True)

F = np.fft.fft2(f)

Fr = np.real(F)
Fi = np.imag(F)

Fs = np.abs(F)
Fa = np.angle(F)

f_out = np.fft.fftshift(np.log(Fs)+1)
plt.imshow(Fa, cmap="gray")
plt.show()