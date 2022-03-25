import numpy as np
from scipy import ndimage as ndi
import imageio
import skimage
from matplotlib import pyplot as plt
from basic_filters import plot_filter

def rnotch_filter(shape, D0=0, angle=0, ftype='ideal', reject=True, W=1, n=1):
    """
    :param shape: shape of the filter
    :param D0: start (from center) of rectangular notch(es) (till image edge). (shape=(K,))
    :param angle: angle (in degree) of notch with x-axis
    :param ftype: type of the filter:'ideal', 'gaussian', or 'butterworth'
    :param reject: True or False (pass filter)
    :param W: notch width(s)
    :param n: order(s) of the butterworth filter notches
    :return: rectangular notch filter in the frequency domain
    """
    r, c = shape
    D0 = np.atleast_1d(D0)
    K = D0.size

    angle = np.ones((K,)) * angle
    W = np.ones((K,)) * W
    n = np.ones((K,)) * n

    d = 2 * np.ceil(np.sqrt((r / 2) ** 2 + (c / 2) ** 2)) + 1
    R, C = [x - d // 2 for x in np.ogrid[:d, :d]]

    H = []
    Hk = []
    Hk = np.zeros([int(d), int(d)])
    center = int(d) // 2
    width = int(W)

    for k in range(K):
        if (ftype == 'ideal'):
            Hk[center-width: center+width+1,0:center-D0[k]+1] = 1.0
            Hk[center-width: center+width+1,center+D0[k]:] = 1.0
            Hk = 1-Hk
        elif (ftype=='gaussian'):
            Hk[:,0:center-D0[k]+1] = np.exp(-R ** 2 / (2 * D0 ** 2))
            Hk[:,center + D0[k]:] = np.exp(-R ** 2 / (2 * D0 ** 2))
            Hk = 1 - Hk

        elif (ftype == 'butterworth'):
            Hk[:, 0:center - D0[k] + 1] = 1 / (1 + (R / D0) ** (2 * n))
            Hk[:, center + D0[k]:] = 1 / (1 + (R / D0) ** (2 * n))
            Hk = 1 - Hk

        Hk = ndi.rotate(Hk, angle=angle[k], mode='reflect', reshape=False, order=1)

        Hk = Hk[int(d // 2 - r // 2):int(d // 2 + r // 2 + r % 2), \
             int(d // 2 - c // 2):int(d // 2 + c // 2 + c % 2)]
        H.append(Hk)

    H = np.array(H).prod(axis=0)

    if not reject:
        H = np.abs(1 - H)
    return H
"""
functie testen
"""
"""
img = imageio.imread('./imgs/imgs/saturn_rings.tif')
img = skimage.util.img_as_float(img)
F = np.fft.fftshift(np.fft.fft2(img))

fig, ax = plt.subplots(3,4)
fig.suptitle('rectangular notch filters', fontsize=16)
ax[0,0].set_title('Noisy image')
ax[0,1].set_title('Fourier spectrum')
ax[0,2].set_title('Filtered Fourier spectrum')
ax[0,3].set_title('Filtered image')
left, width = 0, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
ax[0,0].text(left, 0.5 * (bottom + top), 'ideal',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax[0,0].transAxes)
ax[0,1].text(left, 0.5 * (-2 + top), 'butterworth',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax[0,0].transAxes)
ax[2,0].text(left, 0.5 * (-4.5 + top), 'gaussian',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax[0,0].transAxes)
for i in range(3):
    if(i==0):
        ftype = 'ideal'
    elif(i==1):
        ftype = 'butterworth'
    else:
        ftype = 'gaussian'
    H = rnotch_filter(img.shape,D0=10,ftype=ftype,W=10,angle=90,n=5)
    G = F*H
    g = np.real(np.fft.ifft2(np.fft.fftshift(G)))

    ax[i,0].imshow(img);ax[i,0].axis('off')
    ax[i,1].imshow(np.log(1+np.abs(F)));ax[i,1].axis('off')
    ax[i,2].imshow(np.log(1+np.abs(G)));ax[i,2].axis('off')
    ax[i,3].imshow(g);ax[i,3].axis('off')
plt.show()
"""