import numpy as np
import matplotlib.pyplot as plt
def dftfilt(f, H, pad=False):
    """
    filter the image f in the frequency domain with the filter H (G=F*H) and return the filtered image g
    :param f: input image f
    :param H: filter in the frequency domain (centered at image center)
    :param pad: using padding before filtering (if pad = True: H.shape should be 2*image.shape
    return: filtered image g
    """
    if pad:
        f = np.pad(f, ((f.shape[0]//2), (0,  f.shape[1])), 'constant')
        F = np.fft.fftshift(np.fft.fft2(f))
        G = F * H
        g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
        g = g[:f.shape[0]//2, :f.shape[1]//2]
    else:
        F = np.fft.fftshift(np.fft.fft2(f))
        G = F * H
        g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

#MOET NOG GECENTERD WORDEN
    return g
