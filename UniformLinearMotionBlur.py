import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.transform import rotate
from skimage import util, io, color
from numpy.fft import fft2, ifft2, fftshift


def crop2center(psf, shape, d):
    r, c = shape
    return psf[int(d // 2 - r // 2):int(d // 2 + r // 2 + r % 2), int(d // 2 - c // 2):int(d // 2 + c // 2 + c % 2)]


def linearmotionblurfilter(shape, length=1, angle=0, domain='spatial'):
    r, c = shape
    center= (r//2,c//2)
    d = int(2 * np.ceil(np.sqrt((r / 2) ** 2 + (c / 2) ** 2)) + 1)  # diagonal size+1
    psf = np.zeros((d, d))
    psf[d // 2, d // 2:d // 2 + length] = 1

    psf = skimage.transform.rotate(psf, angle,  mode='symmetric')

    psf = crop2center(psf, shape, d)
    psf = np.fft.fftshift(psf)
    if (domain == 'freq') | (domain == 'frequency'):
        psf = np.fft.fftshift(np.fft.fft2(psf))
    return psf


# Main
# Hlm filter
img = color.rgb2gray(io.imread('imgs\daenerys.jpg'))
img[300, 600] = 0
a, b, T = 0.02, 0.02, 1
r, c = img.shape
R, C = np.ogrid[:r, :c]
x = (a * (R - r // 2) + b * (C - c // 2)) * np.pi + 1e-16
Hlm = T / x * np.sin(x) * np.exp(-1j * x)

# Linearmotionblurfilter
psf_filter = linearmotionblurfilter(Hlm.shape, length=20, angle=-20, domain='spatial')
plt.imshow(np.real(np.fft.fftshift(psf_filter)), cmap='gray')
plt.show()

# Hoe moet de PSF eruit zien?
psf = np.real(fftshift(ifft2(fftshift(Hlm))))
plt.imshow(psf, cmap='gray')
plt.show()
