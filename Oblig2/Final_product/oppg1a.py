from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

def conv2(f, filter):
    f_con = convolve2d(f, filter)
    #f_con = convolve2d(f, filter, "same")
    
    fig1, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
    ax2.imshow(f_con, cmap="gray", vmin=0, vmax=255)
    plt.title('Med konvolusjon')

def fft2(f, filter):
    

    FN, FM = filter.shape
    #Legger til padding slik at filteret har samme dimensjoner som bildet
    """top = 0
    bottom = N-FN
    left = 0
    right = M-FM
    filter_padded = np.pad(filter, ((top,bottom),(left,right)), 'constant')"""

    filter_fft = np.fft.fft2(filter, (N,M))

    #Sender bådet bildet og filteret inn i frekvensdomenet og multipliserer de    
    #filter_fft = np.fft.fft2(filter_padded)
    f_fft = np.fft.fft2(f)
    f_out = f_fft*filter_fft
    #Returnerer produktet til bildedomenet
    f_out = (np.fft.ifft2(f_out))

    fig2, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(f, cmap="gray", vmin=0, vmax=255)
    ax2.imshow(f_out.real, cmap="gray", vmin=0, vmax=255)
    plt.title('Med fft')




filter = np.array([[1/(15*15)]*15]*15)
f = imread('cow.png', as_gray=True)
N,M = f.shape

conv2(f, filter)
fft2(f, filter)

plt.show()

