import numpy as np


def periodic_noise(shape, C, A=None, B=None):
    """
    generate periodic noise arrays(r:spatial domain,R:frequency domain)
    shape: shape of array
    C: (K,2) array of energy burst frequencies
    A: (K,) vector of burst magnitudes
    B: (K,2) array of phase shifts(Bx,By)


        """
    "--------------------------Spatial domain--------------------------"
    M = shape[0]
    N = shape[1]
    x, y = np.mgrid[:M, :N].astype('float32')
    r = np.zeros(shape)
    if A is None: A = np.ones(C.shape[0])
    if B is None: B = np.zeros((C.shape[0], 2))

    for i in range(C.shape[0]):
        r = A[i] * np.sin(2 * np.pi * (C[i, 0] * (x + B[i, 0]) / M + C[i, 1] * (y + B[i, 1]) / N))

    R = np.fft.fftshift(np.fft.fft2(r))
    return r, R
