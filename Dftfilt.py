
def dftfilt(f, H, pad=False):
    """
    filter the image f in the frequency domain with the filter H (G=F*H) and return the filtered image g
    :param f: input image f
    :param H: filter in the frequency domain (centered at image center)
    :param pad: using padding before filtering (if pad = True: H.shape should be 2*image.shape
    return: filtered image g
    """

    g = f
    return g