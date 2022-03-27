from numpy.fft import fft2,ifft2,fftshift
import numpy as np
import imageio as io
from matplotlib import pyplot as plt
import cv2
from skimage.morphology import square, dilation
from basic_filters import lpcfilter, hpcfilter, plot_filter
import cv2 as cv
import matplotlib.pyplot as plt
import pylab as p
from skimage import io
from skimage import util
import imageio as iio
import numpy as np
from Cband_filter import cband_filter
from skimage.morphology import square, dilation
from PeriodicNoise import periodic_noise


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
    H2 = hpcfilter(shape, D0=D0, n=n, ftype=ftype, center=(centers[0][0]*(-1),centers[0][1]*(-1)))
    H = H1*H2
    for element in centers[1:]:
        H_single_pos = hpcfilter(shape,D0=D0,n=n,ftype=ftype, center=element)
        H_single_neg = hpcfilter(shape, D0=D0,n=n, ftype=ftype, center=(element[0]*(-1),element[1]*(-1)))
        H = H*H_single_pos*H_single_neg
    if(reject):
        return H
    else:
        return 1-H

"""
functie testen
"""

img = iio.imread('./imgs/imgs/SEMintegratedcircuit.jpg')
img = util.img_as_float(img)

M, N = img.shape
thetas = np.array([0,60,120])


r, R = periodic_noise(img.shape, thetas)
g = img + r/3
plt.figure();plt.axis('off');plt.imshow(g, cmap='gray')
plt.show()

G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1+np.abs(G)), square(3))
plt.figure();plt.axis('off');plt.imshow(Gd, cmap='gray')

xy = np.array(plt.ginput(-1, show_clicks=True))
plt.show()
rc = xy[:,::-1]

center=np.array(img.shape)//2
rc = rc-center
D0 = (np.sum(rc**2,axis=1)**0.5).mean()
H = cnotch_filter(img.shape,rc, ftype='ideal',reject=True,D0=3, n=5)
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
