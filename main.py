# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
from skimage import util
import imageio as iio
import numpy as np
from Cband_filter import cband_filter
from skimage.morphology import square, dilation
from PeriodicNoise import periodic_noise
from Dftfilt import dftfilt
from basic_filters import lpcfilter
from Functions import plot_figures
"--------------------------Adaptive median spatial filterin-------------------------"
#TODO

"--------------------------Periodic noise-------------------------"
img = iio.imread(
    'imgs/Apollo17boulder.tif')
img = util.img_as_float(img)

M, N = img.shape
thetas = np.array([[-20, -20], [-20, 20]])
r, R = periodic_noise(img.shape, thetas)
g = img + r / 3

"--------------------------Cband-Filter-------------------------"
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

"--------------------------Cnotch-Filter-------------------------"

#TODO

"--------------------------Rnotch-Filter-------------------------"

#TODO

"--------------------------dftfilt-------------------------"

img = iio.imread('/Users/stefgielen/Documents/school 2021-2022/SEM2/image processing/Oefeningen/imgs/obelix.tif')
img = util.img_as_float(img)
    #filter zonder padding
r, c = img.shape
H = lpcfilter((r, c), ftype='gaussian', D0=30)
g = dftfilt(img, H)
    #filter met padding
Hp = lpcfilter((2 * r, 2 * c), ftype='gaussian', D0=2 * 30)
gp = dftfilt(img, Hp, pad=True)

dftPlots = []
dftTitles = []
dftPlots.append(img); dftTitles.append('original')
dftPlots.append(g); dftTitles.append('gaussian filter no padding')
dftPlots.append(gp); dftTitles.append('gaussian filter with padding')
plot_figures('Dft filter function', np.array(dftPlots), dftTitles, rowSize=1)

"--------------------------Restoration-Filters-------------------------"

#TODO

"--------------------------linearmotionblurfilter-------------------------"

#TODO
