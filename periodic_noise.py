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

    D = 30 * np.sqrt(2)
    u = np.array((D * np.cos(C * np.pi / 180) + M // 2).astype(int))
    v = (D * np.sin(C * np.pi / 180) + N // 2).astype(int)

    if A == None: A = np.ones(C.shape)
    if B == None: B = np.zeros(C.size)

    R = np.zeros(shape, dtype=complex)

    for i in range(C.size):
        R[u, v] = 1j * (A[i] / 2) * M * N * np.exp(0)
    r = np.real(np.fft.ifft2(np.fft.ifftshift(R)))
    return r, R


