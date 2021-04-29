from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

f = imread('lena.png', as_gray=True)


#filter = np.array([[1, 2, 4, 6, 4, 2, 1], [2, 4, 6, 12, 6, 4, 2], [4, 6, 8 ,12, 8, 6, 4], [6, 12, 18 ,24, 18, 12, 6], [4, 6, 8 ,12, 8, 6, 4], [2, 4, 6, 12, 6, 4, 2], [1, 2, 4, 6, 4, 2, 1]])
filterGaus = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
filterX = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
filtery = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])




def addPadding(f):
    N,M = f.shape
    f_exp = np.zeros((N+2,M+2))

    #legger inn-bildet inn i det større bilde
    for i in range(N):
        for j in range(M):
            f_exp[i+1, j+1] = f[i,j]
    #finner nærmeste nabo
    #for x in range(4):
    for i in range(1, N+1):
        f_exp[i,0] = f[i-1,0]
        f_exp[i,M+1] = f[i-1, M-1]
    for i in range(1,M+1):
        f_exp[0,i] = f[0, i-1]
        f_exp[N+1,i] = f[N-1, i-1]
    #legger inn hjørnene
    f_exp[0,0] = f[0,0]
    f_exp[0,M+1] = f[0,M-1]
    f_exp[N+1,0] = f[N-1,0]
    f_exp[N+1,M+1] = f[N-1,M-1]

    return f_exp

def addPadding2(f, filter_size):
    N,M = f.shape
    margin = math.floor(filter_size/2)
    f_exp = np.zeros((N+margin*2,M+margin*2))
    N_1,M_1 = f_exp.shape
    #legger inn-bildet inn i det større bilde
    for i in range(N):
        for j in range(M):
            f_exp[i+margin, j+margin] = f[i,j]
    #finner nærmeste nabo

    for j in range(margin-1, -1, -1):
        for i in range(margin, N+margin):
            f_exp[i,j] = f[i-margin,0]
            f_exp[i,M_1-j-1] = f[i-margin, M-1]
        for i in range(margin, M+margin):
            f_exp[j,i] = f[0, i-margin]
            f_exp[N_1-j-1,i] = f[N-1, i-margin]
    #legger inn hjørnene
    for i in range(margin):
        for j in range(margin):
            f_exp[i,j] = f[0,0]
            f_exp[i, M+margin+j] = f[0, M-1]
            f_exp[N+margin+i, j] = f[N-1, 0]
            f_exp[N+margin+i, M+margin+j] = f[N-1, M-1]

    return f_exp

def applyConvolution(f, filter):
    N,M = f.shape
    X,Y = filter.shape
    margin = math.floor(X/2)
    f_out = np.zeros((N-margin*2,M-margin*2))
    for i in range(margin,N-margin):
        for j in range(margin,M-margin):
            sum = 0
            for x in range(-margin, margin+1):
                for y in range(-margin, margin+1):
                    sum += f[i+x, j+y] * filter[x, y]
            f_out[i-margin, j-margin] = sum/(X*Y)

    return f_out


def oneD_filters(img):
    Kx = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    Ky = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])

    Ix = applyConvolution(img, Kx)
    Iy = applyConvolution(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

#Finner nermeseste
def closest_dir_function(grad_dir) :
    closest_dir_arr = np.zeros(grad_dir.shape)
    for i in range(1, int(grad_dir.shape[0] - 1)) :
        for j in range(1, int(grad_dir.shape[1] - 1)) :

            if((grad_dir[i, j] > -22.5 and grad_dir[i, j] <= 22.5) or (grad_dir[i, j] <= -157.5 and grad_dir[i, j] > 157.5)) :
                closest_dir_arr[i, j] = 0

            elif((grad_dir[i, j] > 22.5 and grad_dir[i, j] <= 67.5) or (grad_dir[i, j] <= -112.5 and grad_dir[i, j] > -157.5)) :
                closest_dir_arr[i, j] = 45

            elif((grad_dir[i, j] > 67.5 and grad_dir[i, j] <= 112.5) or (grad_dir[i, j] <= -67.5 and grad_dir[i, j] > -112.5)) :
                closest_dir_arr[i, j] = 90

            else:
                closest_dir_arr[i, j] = 135

    return closest_dir_arr


#2.b : Convert to thinned edge
def non_maximal_suppressor(grad_mag, closest_dir) :
    thinned_output = np.zeros(grad_mag.shape)
    for i in range(1, int(grad_mag.shape[0] - 1)) :
        for j in range(1, int(grad_mag.shape[1] - 1)) :

            if(closest_dir[i, j] == 0) :
                if((grad_mag[i, j] > grad_mag[i, j+1]) and (grad_mag[i, j] > grad_mag[i, j-1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0

            elif(closest_dir[i, j] == 45) :
                if((grad_mag[i, j] > grad_mag[i+1, j+1]) and (grad_mag[i, j] > grad_mag[i-1, j-1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0

            elif(closest_dir[i, j] == 90) :
                if((grad_mag[i, j] > grad_mag[i+1, j]) and (grad_mag[i, j] > grad_mag[i-1, j])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0

            else :
                if((grad_mag[i, j] > grad_mag[i+1, j-1]) and (grad_mag[i, j] > grad_mag[i-1, j+1])) :
                    thinned_output[i, j] = grad_mag[i, j]
                else :
                    thinned_output[i, j] = 0

    return thinned_output/np.max(thinned_output)

#def hysterese(img, Tl, Th)

def makeGaus(filter_size, std):
    filter = np.zeros((filter_size, filter_size))
    for x in range(filter_size):
        for y in range(filter_size):
            filter[x,y] = 1**(-(x**2+y**2)/(2*std))
    return filter





#f_exp = addPadding2(f, filter.shape[0])
filterGaus = makeGaus(3, np.sqrt(np.var(f)))
f_exp = addPadding(f)

#f_exp = applyConvolution (f, filterGaus)
#G, theta = oneD_filters(f_exp)
#closest = closest_dir_function(G)
#f_out = non_maximal_suppressor(G, closest)



#f_out = applyConvolution(f_exp, filterGaus)




"""plt.imshow(G, cmap="gray", vmin=0, vmax=255)
plt.title('Gradient')
plt.figure()"""
plt.imshow(f_exp, cmap="gray")
plt.title('Gradient - mindre kanter')
plt.show()
