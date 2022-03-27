from scipy import signal as sig
import numpy as np
import imageio as io
from matplotlib import pyplot as plt
from numpy.fft import (fft2,ifft2,fftshift)


def lpcfilter(shape, ftype='ideal', D0=0, n=1, center=(0, 0)):
    """
    Generate low pass circular filter H in the frequency domain
    ftype: tupe of the filter: 'ideal', 'gaussian', or 'butterworth',
    reject: True or False (pass filter),
    D0: band radius,
    n: order of the butterworth filter,
    center: center of the filter
    """
    r, c = shape
    H = np.zeros((r, c))
    R, C = np.ogrid[:r, :c]
    R0 = D0
    D = np.sqrt((R - (r // 2 + center[0])) ** 2 + (C - (c // 2 + center[1])) ** 2)
    if (ftype == 'ideal'):
        H[D < R0] = 1.0
        return H
    elif (ftype == 'butterworth'):
        H = 1 / (1 + (D / R0) ** (2 * n))
        return H
    elif (ftype == 'gaussian'):
        H = np.exp(-D ** 2 / (2 * D0 ** 2))
        return H


def plot_filter(H):
    """
    plot the filter in the frequency domain on a 3D surface plot
    """
    x = np.arange(start=0, stop=len(H[:, 0]))
    y = np.arange(start=0, stop=len(H[0, :]))
    X, Y = np.meshgrid(y, x)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(Y, X, H)


from scipy import signal as sig
import numpy as np


def hpcfilter(shape, ftype='ideal', D0=0, n=1, center=(0, 0)):
    """
    Generate high pass circular filter H in the frequency domain
    ftype: tupe of the filter: 'ideal', 'gaussian', or 'butterworth',
    reject: True or False (pass filter),
    D0: band radius,
    n: order of the butterworth filter,
    center: center of the filter
    """
    return 1 - lpcfilter(shape, ftype, D0, n, center)


def plot_filter(H):
    x = np.arange(start=0, stop=len(H[:, 0]))
    y = np.arange(start=0, stop=len(H[0, :]))
    X, Y = np.meshgrid(y, x)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Y, X, H)

"""
Testing the filters
"""
"""
img = io.imread('./imgs/imgs/SEMintegratedcircuit.jpg')
H = fftshift(hpcfilter(shape = img.shape,D0=10,n=5,ftype = 'gaussian'))
#h = fftshift(h)
F = fft2(img)
#H = fft2(h)
#plot_filter(fftshift(H))
G = F*H
g = np.real(ifft2(G))
plt.figure()
#plt.imshow(H)
#plt.imshow(F)
plt.imshow(g, cmap='gray')
plt.show()
"""