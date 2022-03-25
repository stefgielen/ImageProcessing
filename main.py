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
from Dftfilt import dftfilt
from basic_filters import lpcfilter

"--------------------------Periodic noise-------------------------"




"""
img = iio.imread('/Users/stefgielen/Documents/school 2021-2022/SEM2/image processing/Oefeningen/imgs/Apollo17boulder.tif')
img = util.img_as_float(img)

"--------------------------Periodic noise-------------------------"

M, N = img.shape
thetas = np.array([[-20, -20], [24, 44], [-20, 20], [24, -44]])
r, R = periodic_noise(img.shape, thetas)
g = img + r/3

"--------------------------Cband-Filter-------------------------"

G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1+np.abs(G)), square(3))
plt.figure();plt.axis('off');plt.imshow(Gd, cmap='gray')

xy = np.array(plt.ginput(-1, show_clicks=True))
plt.show()
rc = xy[:, ::-1]

center = np.array(img.shape)//2
rc = rc-center
D0 = (np.sum(rc**2,axis=1)**0.5).mean()
H = cband_filter(img.shape, D0, ftype='gaussian',reject=True,W=3)
G2 = G*H
g2 = np.real(np.fft.ifft2(np.fft.fftshift(G2)))

plt.plot();plt.axis('off');plt.imshow(g, cmap='gray')
plt.show()
plt.plot();plt.axis('off');plt.imshow(Gd, cmap='gray')
plt.show()
plt.plot();plt.axis('off');plt.imshow(H, cmap='gray')
plt.show()
plt.plot();plt.axis('off');plt.imshow(g2, cmap='gray')
plt.show()
"""
"--------------------------dftfilt-------------------------"

img = iio.imread('/Users/stefgielen/Documents/school 2021-2022/SEM2/image processing/Oefeningen/imgs/obelix.tif')
img = util.img_as_float(img)

r, c = img.shape
H = lpcfilter((r, c), ftype='gaussian', D0=30)
g = dftfilt(img, H)

Hp = lpcfilter((2*r, 2*c), ftype='gaussian', D0=2*30)
gp = dftfilt(img, Hp, pad=True)

plt.plot(); plt.axis('off'); plt.imshow(img, cmap='gray')
plt.show()
plt.plot(); plt.axis('off'); plt.imshow(g, cmap='gray')
plt.show()
plt.plot(); plt.axis('off'); plt.imshow(gp, cmap='gray')
plt.show()



