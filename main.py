# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import matplotlib.pyplot as plt
import pylab as p
from skimage import io
from skimage import util
import imageio as iio
import numpy as np
from cband_filter import cband_filter
from skimage.morphology import square, dilation
from periodic_noise import periodic_noise

"--------------------------Periodic noise-------------------------"



"""
img = iio.imread('/Users/stefgielen/Documents/school 2021-2022/SEM2/image processing/Oefeningen/imgs/Apollo17boulder.tif')
img = util.img_as_float(img)
M, N = img.shape
thetas = np.array([0,60,120])

D = 30 * np.sqrt(2)
u =np.array((D * np.cos(thetas * np.pi / 180) + M // 2).astype(int))
v = (D * np.sin(thetas * np.pi / 180) + N // 2).astype(int)

C = np.concatenate((u.reshape(3,1),v.reshape(3,1)),axis=1)
print(C)
print(C.shape)
A = np.array([1/3,1,3])    #amplitudes -> nu op 1/3 gezet 1-By-K matrix

Bx = np.array([0,0,0])
By = np.array([0,0,0])
B = np.concatenate((Bx.reshape(3,1),By.reshape(3,1)),axis=1) #K-by-2 matrix
print(B)
r = periodic_noise(img.shape, C, A, B)

cv.imshow('1',img + r[0])
cv.waitKey(0)

cv.imshow('1',img + r[1])
cv.waitKey(0)

cv.imshow('1',img + r[2])
cv.waitKey(0)
"""
"--------------------------Periodic noise-------------------------"
"""
img = iio.imread('/Users/stefgielen/Documents/school 2021-2022/SEM2/image processing/Oefeningen/imgs/Apollo17boulder.tif')
img = util.img_as_float(img)

M, N = img.shape
thetas = np.array([0,60,120])


r, R = periodic_noise(img.shape, thetas)
g = img + r/3
#plt.figure();plt.axis('off');plt.imshow(g, cmap='gray')

G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1+np.abs(G)), square(3))
plt.figure();plt.axis('off');plt.imshow(Gd, cmap='gray')

xy = np.array(plt.ginput(-1, show_clicks=True))
#plt.show()
rc = xy[:,::-1]

center=np.array(img.shape)//2
rc = rc-center
D0 = (np.sum(rc**2,axis=1)**0.5).mean()
H = cband_filter(img.shape, D0, ftype='gaussian',reject=True,W=3)
print(H)
G2 = G*H
g2 = np.real(np.fft.ifft2(np.fft.fftshift(G2)))

plt.plot();plt.axis('off');plt.imshow(g)
plt.show()
plt.plot();plt.axis('off');plt.imshow(Gd)
plt.show()
plt.plot();plt.axis('off');plt.imshow(H)
plt.show()
plt.plot();plt.axis('off');plt.imshow(g2)
plt.show()
"""
print("hallokes")