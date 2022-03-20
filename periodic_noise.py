
def periodic_noise(shape, C, A=None, B=None):
    """
    generate periodic noise arrays(r:spatial domain,R:frequency domain)
    shape: shape of array
    C: (K,2) array of energy burst frequencies
    A: (K,) vector of burst magnitudes
    B: (K,2) array of phase shifts(Bx,By)


        """
    "--------------------------Spatial domain--------------------------"
    L = C.shape[0]

    M = shape[0]
    N = shape[1]


    x, y = np.mgrid[:M, :N].astype('float32')
    u0, v0 = 30, 30
    r = np.zeros(C.size)

    for i in range(L):

        r= A[i] * np.sin(2 * np.pi * (C[i, 0] * (x + B[i, 0]) / M + C[i, 1] * (y + B[i, 1]) / N))
        print(r[i])
        print('\n')

    return r