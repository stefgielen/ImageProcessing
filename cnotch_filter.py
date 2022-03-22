from numpy.fft import fft2,ifft2,fftshift
import numpy as np
import imageio as io
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import square, dilation
from basic_filters import lpcfilter, hpcfilter, plot_filter


def cnotch_filter(shape=None, centers=None, ftype='ideal', reject=True, D0=0, n=1):
    """
    Generate circular notch filter in the frequency domain
    shape: shape of the filter,
    centers: notch frequency coordinates (shape = (K,2)),
    ftype: tupe of the filter: 'ideal', 'gaussian', or 'butterworth',
    reject: True or False (pass filter),
    D0: notch size(s),
    n: order(s) of the butterworth filter notches
    """
    H1 = hpcfilter(shape, D0=D0, n=n, ftype=ftype, center=centers[0])
    H2 = hpcfilter(shape, D0=D0, n=n, ftype=ftype, center=(centers[0][0]*(-1),centers[1][0]*(-1)))
    H = H1*H2
    if(ftype == 'butterworth'):
        for element in centers[1:]:
            H_single_pos = hpcfilter(shape,D0=D0,n=n,ftype=ftype, center=element)
            H_single_neg = hpcfilter(shape, D0=D0,n=n, ftype=ftype, center=(element[0]*(-1),element[1]*(-1)))
            H = H*H_single_pos*H_single_neg
        return H

"""
functie testen
"""
from periodic_noise import periodic_noise
img = io.imread('./imgs/imgs/SEMintegratedcircuit.jpg')
M, N = img.shape
thetas = np.array([0,60,120])

r, R = periodic_noise(img.shape, thetas)
g = img + r/3
G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1+np.abs(G)), square(3))
xy = np.array(plt.ginput(-1, show_clicks=True))
plt.show()
rc = xy[:,::-1]

center = np.array(img.shape)//2
rc = rc - center
H = cnotch_filter(img.shape,ftype='butterworth',reject=True,D0=20, centers=[(20,30),(30,40),(100,100),(0,200)])
G2 = G*H
g2 = np.real(ifft2(np.fft.fftshift(G2)))

#plt.imshow(H)
plt.subplot(411);plt.axis('off');plt.imshow(g)
plt.show()
plt.subplot(412);plt.axis('off');plt.imshow(Gd)
plt.show()
plt.subplot(413);plt.axis('off');plt.imshow(H)
plt.show()
plt.subplot(414);plt.axis('off');plt.imshow(g2)
plt.show()
