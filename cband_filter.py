from scipy import signal as sig
import numpy as np


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
        print(D)
        D[D == 0] = 1  # nulwaarde in midden op 1 foefelen
        H = 1 - np.exp(-(((D ** 2) - (D0 ** 2)) / (D * W)) ** 2)
    if not reject:
        H = 1 - H
    return H
