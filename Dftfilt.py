import numpy as np


def dftfilt(f, H, pad=False):
    """
    filter the image f in the frequency domain with the filter H (G=F*H) and return the filtered image g
    :param f: input image f
    :param H: filter in the frequency domain (centered at image center)
    :param pad: using padding before filtering (if pad = True: H.shape should be 2*image.shape
    return: filtered image g
    """
    r, c = f.shape
    if pad:
        # padding voor en na foto, in totaal r padding toegevoegd
        f = np.pad(f, ((r // 2, r // 2), (c // 2, c // 2)), 'constant')
        F = np.fft.fftshift(np.fft.fft2(f))
        G = F * H
        g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
        # croppen naar otiginele size
        g = g[r // 2:3 * (r // 2), c // 2:3 * (c // 2)]
    else:
        F = np.fft.fftshift(np.fft.fft2(f))
        G = F * H
        g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    return g
