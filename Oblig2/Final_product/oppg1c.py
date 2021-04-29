from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import time

def conv2():
    list = [0]
    for i in range(1,filter_numb*2 + 1,2):
        start = time.time()
        filter = np.array([[1/(i*i)]*i]*i)
        convolve2d(f, filter)
        end = time.time()
        list.append(end-start)
    print("conv2 done")
    return list

def fft2():
    list = [0]
    for i in range(1,filter_numb*2,2):
        start = time.time()
        filter = np.array([[1/(i*i)]*i]*i)
        filter_fft = np.fft.fft2(filter, (N,M))
        f_fft = np.fft.fft2(f)
        f_out=f_fft*filter_fft
        np.abs(np.fft.ifft2(f_out))
        end = time.time()
        list.append(end-start)
    print("fft2 done")
    return list

def make_plot(conv2, fft2):
    x_pos = np.arange(len(conv2))
    plt.plot(x_pos, conv2, color = "red")
    plt.plot(x_pos, fft2, color="blue")
    plt.legend(["conv2", "fft2"])
    plt.ylabel('Run time')
    plt.xlabel('Filter size')
    plt.grid()
    plt.savefig("oppg1-plott.png")
    plt.show()


f = imread('cow.png', as_gray=True)
filter_numb = 10
N,M = f.shape
conv2_time = conv2()
fft2_time = fft2()

make_plot(conv2_time, fft2_time)

