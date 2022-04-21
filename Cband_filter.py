import numpy as np
import matplotlib.pyplot as plt
from skimage import util
import imageio as iio

from PeriodicNoise import periodic_noise
from Functions import plot_figures
from skimage.morphology import square, dilation


def cband_filter(shape, D0, ftype='ideal', reject=True, W=1, n=1):
    """
    generate circular band filter in frequency domain
        shape: shape of filter
        D0: band radius or radii(shape = (K,)),
        ftype: type of the filter: ideal, gaussian or butterworth
        reject: True of False (pass filter)
        W= band width(s)
        n:order(s) of the butterworth notches
    """
    H = np.zeros(shape)
    r, c = shape
    R, C = np.ogrid[:r, :c]
    if ftype == 'ideal':
        H[np.sqrt((R - r // 2) ** 2 + (C - c // 2) ** 2) < D0 - W / 2] = 1
        H[np.sqrt((R - r // 2) ** 2 + (C - c // 2) ** 2) > D0 + W / 2] = 1
    elif ftype == 'butterworth':
        D = np.sqrt((R - r // 2) ** 2 + (C - c // 2) ** 2)
        H = 1 / (1 + ((D * W) / (D ** 2 - D0 ** 2)) ** (2 * n))
    elif ftype == 'gaussian':
        D = np.sqrt((R - r // 2) ** 2 + (C - c // 2) ** 2)
        D[D == 0] = 1
        H = 1 - np.exp(-(((D ** 2) - (D0 ** 2)) / (D * W)) ** 2)
    if not reject:
        H = 1 - H
    return H


"--------------------------testcode-------------------------"
"""img = iio.imread('imgs/Apollo17boulder.tif')
img = util.img_as_float(img)

M, N = img.shape
thetas = np.array([[-20, -20], [-20, 20]])
r, R = periodic_noise(img.shape, thetas)
g = img + r / 3

cbandPlots = []
cbandTitles = []
    #Energy bursts weergeven
G = np.fft.fftshift(np.fft.fft2(g))
Gd = dilation(np.log(1 + np.abs(G)), square(3))
plt.figure(); plt.axis('off'); plt.imshow(Gd, cmap='gray')
    #Punten aanduiden
xy = np.array(plt.ginput(-1, show_clicks=True))
rc = xy[:, ::-1]

    #Filters aanmaken en weergeven
center = np.array(img.shape) // 2
rc = rc - center
D0 = (np.sum(rc ** 2, axis=1) ** 0.5).mean()
filters = ['ideal', 'butterworth', 'gaussian']
for k in filters:
    H = cband_filter(img.shape, D0, ftype=k, reject=True, W=10)
    G2 = G * H
    g2 = np.real(np.fft.ifft2(np.fft.fftshift(G2)))
    cbandPlots.append(g); cbandTitles.append('original')
    cbandPlots.append(H); cbandTitles.append(k + ' filter')
    cbandPlots.append(g2); cbandTitles.append(k + ' filtered image')
plot_figures('Cband filter', np.array(cbandPlots), cbandTitles, rowSize=3)
"""