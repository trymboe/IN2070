import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d

f = imread('car.png', as_gray=True)
N,M = f.shape

# Lager filter
h = 1/49 * np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1]])

g = convolve2d(f, h, mode='same') # vanlig konvolusjon

H = fft2(h) # fouriertransformerer filteret

pad_h = np.zeros((N,M))
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        pad_h[i,j] = h[i,j]

# pad_filt[0 : var, 0 : var] = h[0,0]
H1 = fft2(pad_h)

fig = plt.figure('Filtre')
fig.add_subplot(1, 3, 1)
plt.title('Feil størrelse')
plt.imshow(fftshift(np.log(abs(H) + 1)))

fig.add_subplot(1, 3, 2)
plt.title('Versjon 1')
plt.imshow(fftshift(np.log(abs(H1) + 1)))

H2 = fft2(h, (N,M))

fig.add_subplot(1, 3, 3)
plt.title('Versjon 2')
plt.imshow(fftshift(np.log(abs(H2) + 1)))


F = fft2(f) # Fouriertransformerer innbildet
G = F * H1 # Konvolerer
new_g = np.real(ifft2(G)) # Inverstransformerer

test = fft2(new_g)

fig1 = plt.figure('Konvolusjon i bildedomenet')
fig1.add_subplot(1, 2, 1)
plt.imshow(f, cmap='gray', vmin=0, vmax=255)

fig1.add_subplot(1, 2, 2)
plt.imshow(g, cmap='gray', vmin=0, vmax=255)

fig2 = plt.figure('Transformert')

fig2.add_subplot(1, 2, 1)
plt.title('Fourier innbildet')
plt.imshow(fftshift(np.log(abs(F) + 1)))

fig2.add_subplot(1, 2, 2)
plt.title('Fourier filter')
plt.imshow(fftshift(np.log(abs(H1) + 1)))

fig3 = plt.figure('Konvolusjon i frekvensdomenet')
fig3.add_subplot(1, 2, 1)
plt.title('Konvolert innbilde')
plt.imshow(ifftshift(np.log(abs(G) + 1)))

fig3.add_subplot(1, 2, 2)
plt.imshow(new_g, cmap='gray', vmin=0, vmax=255)

plt.show()