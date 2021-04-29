from imageio import imread
import matplotlib.pyplot as plt
import numpy as np


def jpeg(img, q):
    N, M = img.shape
    print("\nCompressing with q =", q)
    img_reduced = img-128
    #Q-matrisen som brukes
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [
                 18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

    #q*Q sendes inn sammen med bilde i en DCT-funksjon
    qQ = q*Q
    F = DCT(img_reduced, qQ)
    #Når bilde er komprimert sendes komprisjonen og originalen inn til en funksjon som
    #regner på komprisjonen
    calculate_space(img, F)
    
    #Når alt er gjort blir bilde dekomprimert
    f = iDCT(F, qQ)
    img_rev = f+128

    #Det dekomprimerte bilde returneres
    return img_rev


def DCT(f, qQ):
    #Denne funksjonen følger formelen gitt i forelesning for DCT-transformasjon
    N, M = f.shape
    F = np.zeros((N, M))
    for i in range(0, N, 8):
        for j in range(0, M, 8):
            for u in range(8):
                for v in range(8):
                    sum = 0
                    for x in range(8):
                        for y in range(8):
                            sum += f[x+i, y+j] * \
                                np.cos(((2*x+1)*(u)*np.pi)/16) * \
                                np.cos(((2*y+1)*(v)*np.pi)/16)
                    F[u+i, v+j] = np.round((1/4*c(u)*c(v) * sum)/qQ[u, v])
    #print("DCT done")
    #Det transformerte bilde returneres
    return F


def iDCT(F, qQ):
    #Dette bilde følger formelen gitt i forelesning for invers DCT-transformasjon
    N, M = F.shape
    f = np.zeros((N, M))
    blocks = np.zeros((N, M))
    for i in range(0, N, 8):
        for j in range(0, M, 8):
            for x in range(8):
                for y in range(8):
                    sum = 0
                    for u in range(8):
                        for v in range(8):
                            blocks[i+u, j+v] = F[i+u, j+v]*qQ[u, v]
                            sum += c(u)*c(v)*blocks[u+i, v+j] * np.cos(
                                ((2*x+1)*(u)*np.pi)/16) * np.cos(((2*y+1)*(v)*np.pi)/16)
                    f[i+x, j+y] = np.round((1/4 * sum), 0)
    #print("iDCT done")
    #Det dekomrpimerte bilde returneres
    return f


def c(a):
    #Dette er kun en hjelpefunksjon for DCT og iDCT
    if a == 0:
        return 1/np.sqrt(2)
    else:
        return 1


def makeHistogram(f):
    #Funksjonen lager et nokså unøyaktig histogram på mange måter, det man trenger for
    #å regne entropi er kun hvor mange tilfeller av hver pixel som skjer, ikke eksakte
    #pikselverdier. Det ordner denne funksjonen
    N, M = f.shape
    G = 500
    hist = [0] * G
    for i in range(N):
        for j in range(M):
            value = int(f[i, j]+128)
            if(value in range(0, G-1)):
                hist[value] = hist[value]+1
    return hist


def normalisertHist(hist):
    #Det normaliserte histogrammet lages
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p


def calculate_entropi(p):
    #Denne funksjonen følger forelsesningens oppskrift på utregning av entropi
    sum = 0
    for i in range(len(p)):
        if(p[i] != 0):
            sum += p[i]*np.log2(p[i])
    return -1*sum


def calculate_space(img, F):
    #denne metoden tar inn originalbildet og det transformerte bilde og regner ut
    #tall ang komprisjonen

    #her følges formelen for utregning av komprisjonsrate
    entropi_img = calculate_entropi(normalisertHist(makeHistogram(img)))
    entropi_F = calculate_entropi(normalisertHist(makeHistogram(F)))
    comprimation_rate = entropi_img/entropi_F
    print("The compression rate is" , np.round(comprimation_rate))

    N,M = img.shape
    #Relativ redundans
    RR = 1-(entropi_F/entropi_img)
    #relativ redundans brukes for å regne hvor mye som optimalt blir spart
    space_use_for_compressed_image = (1-RR)*N*M

    print("The compressed image can optimaly use", space_use_for_compressed_image, "bytes. A", np.round(RR*100), "percent decrease from uncompressed")



f = imread('uio.png', as_gray=True)

#Her kjøres alle komprisjonene
q = 0.1
img1 = jpeg(f, q)
q = 0.5
img2 = jpeg(f, q)
q = 2
img3 = jpeg(f, q)
q = 8
img4 = jpeg(f, q)
q = 32
img5 = jpeg(f, q)

#Følgende kode plotter bildene mot hverandre
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 3
fig.add_subplot(rows, columns, 1)
  
plt.imshow(f, cmap="gray")
plt.axis('off')
plt.title("Original")
  
fig.add_subplot(rows, columns, 2)
plt.imshow(img1, cmap="gray")
plt.axis('off')
plt.title("q = 0.1")

fig.add_subplot(rows, columns, 3)
plt.imshow(img2, cmap="gray")
plt.axis('off')
plt.title("q = 0.5")
  
fig.add_subplot(rows, columns, 4)
plt.imshow(img3, cmap="gray")
plt.axis('off')
plt.title("q = 2")

fig.add_subplot(rows, columns, 5)
plt.imshow(img4, cmap="gray")
plt.axis('off')
plt.title("q = 8")

fig.add_subplot(rows, columns, 6)
plt.imshow(img5, cmap="gray")
plt.axis('off')
plt.title("q = 32")


#Lagre filer
plt.figure()
plt.imshow(img1, cmap="gray")
plt.title("q = 0.1")
plt.savefig("q=0.1.png")

plt.figure()
plt.imshow(img2, cmap="gray")
plt.title("q = 0.5")
plt.savefig("q=0.5.png")

plt.figure()
plt.imshow(img3, cmap="gray")
plt.title("q = 2")
plt.savefig("q=2.png")

plt.figure()
plt.imshow(img4, cmap="gray")
plt.title("q = 8")
plt.savefig("q=8.png")

plt.figure()
plt.imshow(img5, cmap="gray")
plt.title("q = 32")
plt.savefig("q=32.png")


plt.show()
