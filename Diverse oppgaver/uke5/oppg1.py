from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

G = 256

def lagHistogram(f):
    array = np.zeros(G)
    (N, M) = f.shape
    for i in range(N):
        for j in range(M):
            value = int(f[i,j])
            if(value in range(0, G-1)):
                array[value] = array[value]+1
    y_pos = np.arange(len(array))
    return array


# Load a "clean" image
im_clean = imread('textImage_clean.png', as_gray=True)
im_skrift = imread('skrift.png', as_gray=True)
(N, M) = im_clean.shape
f = np.zeros((N,M))

# Add some noise
noiseStd = 30 # How much white noise to add
im_noisy = im_clean + noiseStd * np.random.normal(0, 1, (N,M))

# Add varying light-intesity model
lightFactor = 100 # Increasing this increases effect of our varying-light model
lightMask = np.array([[(x-M/2)/M for x in range(1, M+1)] for y in range(N)])
im_light = im_clean + lightFactor * lightMask

# Separate background and foreground pixels using our "clean" image
backgroundPixels = [im_noisy[x][y] for x in range(N) for y in range(M) if im_clean[x][y] < 150]
foregroundPixels = [im_noisy[x][y] for x in range(N) for y in range(M) if im_clean[x][y] > 150]

"""terskel = 150

for i in range(M):
    for j in range(N):
        if im_clean[i,j] > terskel:
            f[i,j] = 255
        else:
            f[i,j] = 0"""

def otsus(f):
    (N, M) = f.shape
    f_out = np.zeros((N,M))
    h = lagHistogram(f)

    length = len(h)
    sum = 0
    terskel = 0
    for i in range(length):
        sum += h[i]
        if sum > np.sum(h)/2 and h[i] != h[i+1]:
            terskel = i+1
            break

    sum_v = 0
    sum_h = 0
    if terskel+i < 256 and terskel-i > 0:
        for i in range(10):
            sum_v += h[terskel-i]
            sum_h += h[terskel+i]
    while sum_v > sum_h:
        terskel += 1
        sum_v = 0
        sum_h = 0
        for i in range(10):
            if terskel+i < 256 and terskel-i > 0:
                sum_v += h[terskel-i]
                sum_h += h[terskel+i]
            else:
                break

    terskel -= 5
    print(terskel)
    for i in range(N):
        for j in range(M):
            if f[i,j] > terskel:
                f_out[i,j] = 255
            else:
                f_out[i,j] = 0


    """
    fig2, (ax4) = plt.subplots(1)
    ax4.bar(G, h)

    fig1, (ax1,ax2) = plt.subplots(2)
    ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
    ax2.imshow(f_out, cmap="gray", vmin=0, vmax=255)
    plt.show()"""

    return f_out

def adaptiv(f):
    h = lagHistogram(f)
    intervall = 8
    im = np.zeros((2*intervall, 2*intervall))
    f_out = np.zeros((N,M))
    for x in range(0, N, 2*intervall):
        for y in range(0, N, 2*intervall):
            for i in range(2*intervall):
                for j in range(2*intervall):
                    im[i,j] = f[x+i, y+j]
            im = otsus(im)
            for i in range(2*intervall):
                for j in range(2*intervall):
                    f_out[x+i, y+j] = im[i,j]

    y_pos = np.arange(len(h))
    fig2, (ax4) = plt.subplots(1)
    ax4.bar(G, h)

    fig1, (ax1,ax2) = plt.subplots(2)
    ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
    ax2.imshow(f_out, cmap="gray", vmin=0, vmax=255)
    plt.show()

#adaptiv(im_noisy)
f = otsus(im_noisy)
h_clean = lagHistogram(im_clean)
h_noisy = lagHistogram(im_noisy)
h_light = lagHistogram(im_light)
y_pos = np.arange(len(h_clean))

fig1, (ax1,ax2,ax3) = plt.subplots(3)
ax1.imshow(im_clean, cmap="gray", vmin=0, vmax=255)
ax2.imshow(im_noisy, cmap="gray", vmin=0, vmax=255)
ax3.imshow(f, cmap="gray", vmin=0, vmax=255)

"""plt.figure()
plt.bar(y_pos, foregroundPixels)
plt.figure()
plt.bar(y_pos, backgroundPixels)
"""
fig2, (ax4,ax5,ax6) = plt.subplots(3)
ax4.bar(y_pos, h_clean)
ax5.bar(y_pos, h_noisy)
ax6.bar(y_pos, h_light)

plt.show()
