import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import scipy
from scipy import ndimage
import statistics


# g=figure, Smax= max windowsize (oneven en groter dan 1)
def adpmedian(g, Smax=9):
    s = 3  # windowsize
    bookkeeper = np.ones(g.shape, dtype=bool)  # true als pixel nog niet is aangepast false wnr wel
    f = g.copy()

    for i in range(s, Smax + 1, 2):
        print('windowsize:', i)
        f, bookkeeper = levelA(f, i, bookkeeper)

    # Wanneer windowsize groter dan Smax zou worden:
    zmed = scipy.ndimage.generic_filter(input=g, function=statistics.median, size=Smax, mode='constant')
    f = np.where(bookkeeper, zmed, f)
    return f


def levelA(f, i, bookkeeper):
    zmin = scipy.ndimage.generic_filter(input=f, function=min, size=i, mode='constant')
    zmax = scipy.ndimage.generic_filter(input=f, function=max, size=i, mode='constant')
    zmed = scipy.ndimage.generic_filter(input=f, function=statistics.median, size=i, mode='constant')

    #boolen arrays
    A1_groterdannul = np.greater(zmed, zmin)
    A2_kleinerdannul = np.less(zmed, zmax)
    B1_groterdannul = np.greater(f, zmin)
    B2_kleinerdannul = np.less(f, zmax)

    # LevelB
    levelB_zmed = bookkeeper * A1_groterdannul * A2_kleinerdannul * np.invert(B1_groterdannul) * np.invert(B2_kleinerdannul)
    levelB_zxy = bookkeeper * A1_groterdannul * A2_kleinerdannul * B1_groterdannul * B2_kleinerdannul
    f = np.where(levelB_zmed, zmed, f)
    bookkeeper = np.where(levelB_zmed, False, bookkeeper)
    bookkeeper = np.where(levelB_zxy, False, bookkeeper)
    return f, bookkeeper


# Main
fig = io.imread('imgs\ckt-board-saltpep.tif')
filtered_fig = adpmedian(fig)
plt.imshow(filtered_fig, cmap='gray')
plt.show()
