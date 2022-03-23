import numpy as np
from scipy import ndimage as ndi
import imageio
import skimage
from matplotlib import pyplot as plt

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
    r,c = shape
    D0 = np.atleast_1d(D0)
    K = D0.size

    angle = np.ones((K,))*angle
    W = np.ones((K,))*W
    n = np.ones((K,))*n

    d = 2*np.ceil(np.sqrt((r/2)**2 + (c/2)**2))+1
    R,C = [x-d//2 for x in np.ogrid[:d,:d]]

    H = []
    for k in range(K):
        if (ftype == 'ideal'):
            #TODO: code for horizontal ideal filter
            Hk = np.zeros(shape)

        """
        elif (ftype=='gaussian'):
            #TODO: code for horizontal gaussian filter
        elif (ftype == 'butterworth'):
            #TODO: code for horizontal butterworth filter
        """
        Hk = ndi.rotate(Hk,angle=angle[k],mode='reflect',reshape=False,order=1)

        Hk = Hk[int(d//2-r//2):int(d//2 + r//2 + r%2), \
             int(d//2 - c//2):int(d//2 + c//2 + c%2)]
        H.append(Hk)

    H = np.array(H).prod(axis=0)

    if (reject == False):
        H = np.abs(1-H)
    print(H)
    return H


def rnotch_filter_test(shape, D0=0, angle=0, ftype='ideal', reject=True, W=1, n=1):
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
    for k in range(K):
        if (ftype == 'ideal'):
            # TODO: code for horizontal ideal filter
            Hk = np.zeros([int(d),int(d)])
            center = int(d)//2
            width = int(W)
            Hk[center-width: center+width+1,0:center-D0[k]+1] = 1.0
            Hk[center-width: center+width+1,center+D0[k]:] = 1.0
            Hk = 1-Hk
            #print(Hk)
        """
        elif (ftype=='gaussian'):
            #TODO: code for horizontal gaussian filter
        elif (ftype == 'butterworth'):
            #TODO: code for horizontal butterworth filter
        """
        Hk = ndi.rotate(Hk, angle=angle[k], mode='reflect', reshape=False, order=1)

        Hk = Hk[int(d // 2 - r // 2):int(d // 2 + r // 2 + r % 2), \
             int(d // 2 - c // 2):int(d // 2 + c // 2 + c % 2)]
        H.append(Hk)

    H = np.array(H).prod(axis=0)

    if (reject == False):
        H = np.abs(1 - H)
    return H

#test_filter = rnotch_filter_test((11,11), D0=[2],W=1,angle=90)
#print(test_filter)

img = imageio.imread('./imgs/imgs/saturn_rings.tif')
img = skimage.util.img_as_float(img)
F = np.fft.fftshift(np.fft.fft2(img))
#plt.imshow(np.log(1+np.abs(F)))
plt.show()
H = rnotch_filter_test(img.shape,D0=10,ftype='ideal',W=10,angle=90)
G = F*H
g = np.real(np.fft.ifft2(np.fft.fftshift(G)))

#fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1, figsize=(3,12), sharex=True)
fig, ax = plt.subplots(1,4)
ax[1].imshow(img)
ax[2].imshow(np.log(1+np.abs(F)))
ax[3].imshow(np.log(1+np.abs(G)))
ax[0].imshow(g)
plt.show()

"""
ax1.imshow(img);ax1.axis('off');
ax1.set_title('Noisy image')
plt.show()
ax2.imshow(img);ax2.axis('off')
ax2.set_title('Fourier spectrum')
plt.show()
"""
"""
ax3.imshow(np.log(1+np.abs(G)));ax3.axis('off')
ax3.set_title('Filtered fourier spectrum')
plt.show()
ax4.imshow(g);ax4.axis('off');
ax4.set_title('Filterd image')
plt.show()
"""