import cv2
import imageio
import matplotlib.pyplot as plt

from skimage.transform import ProjectiveTransform, warp, SimilarityTransform
from skimage.feature import (match_descriptors, ORB, plot_matches)
from skimage.measure import ransac

import imageio as iio
import numpy as np
from AffineTransformation import get_tf_model
from Functions import plot_figures
from skimage import color, io

src_im  = iio.imread('imgs/daenerys.jpg')
dst_im  = iio.imread('imgs/times-square.jpg')

r, c = src_im.shape[:2]
src_corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])    #xy formaat

plt.figure(); plt.imshow(dst_im)
dst_corners = np.array(plt.ginput(4, 0))    # corners te vervangen oppervlak
print(dst_corners)

tf_model = ProjectiveTransform()
tf_model.estimate(src_corners, dst_corners)
src_tf_im = warp(src_im, tf_model.inverse)

def _get_stitch_images(im0, im1, tf_model = None, n_keypoints=500,
                       min_samples=4, residual_threshold=2, **kwargs):
    if tf_model is None:
        tf_model = SimilarityTransform()
    elif tf_model == 'auto':
        tf_model = get_tf_model(im1, im0, xTransform=ProjectiveTransform,
                                n_keypoints=n_keypoints, min_samples=min_samples,
                                residual_threshold=residual_threshold, **kwargs)
    r, c = im0.shape[:2]
    corners0 = np.array([[0, 0], [0, r], [c, 0], [c, r]])   # groot genoeg beeld zoeken voor gemergede image
    r, c = im1.shape[:2]
    corners1 = np.array([[0, 0], [0, r], [c, 0], [c, r]])
    wcorners1 = tf_model(corners1)

    all_corners = np.vstack((corners0, corners1, wcorners1))
    min_corner = all_corners.min(axis=0)
    max_corner = all_corners.max(axis=0)
    new_shape = max_corner - min_corner
    new_shape = np.ceil(int(new_shape[::-1]))     # groot genoege output shape

    shift = SimilarityTransform(translation = -min_corner)  # voor positieve coördinaten
    im0_ = warp(im0, shift.inverse, output_shape=new_shape, cval=-1)
    im1_ = warp(im1, (tf_model+shift).inverse, output_shape=new_shape, cval=-1)     # cval toont welke pixels achtergrond
    return im0_, im1_

def _merge_stitch_images(im0_,im1_,mask_idx=None,cval=0):
    if mask_idx is not None:
        if mask_idx==0:
            im1_[im0_>-1] = 0
        elif mask_idx==1:
            im0_[im1_>-1] = 0       # vervang overlappende pixels -> mask creeren van 1 figuur

    else:
        alpha = 1.0*(im0_[:, :, 0] != -1) + 1.0*(im1_[:, :, 0] != -1)   # average overlappende pixels
    bgmask = -1*((im0_==-1)&(im1_==-1)) # mask die onthoudt wat achtergrond is
    im0_[im0_ == -1] = 0
    im1_[im1_ == -1] = 0                # achtergrond 0 zetten
    merged = im0_ + im1_                # voeg figuren samen
    if mask_idx is None: merged /=np.maximum(alpha, 1)[..., None]   #intensiteit van gesomde pixels terugschalen
    merged[bgmask==-1] = cval #achtergrond van beide images naar cval zetten
    return merged


src_im  = iio.imread('imgs/daenerys.jpg')
dst_im  = iio.imread('imgs/times-square.jpg')

r, c = src_im.shape[:2]
src_corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])    #xy formaat

plt.figure(); plt.imshow(dst_im)
dst_corners = np.array(plt.ginput(4, 0))    # corners te vervangen oppervlak
print(dst_corners)

tf_model = ProjectiveTransform()
tf_model.estimate(src_corners, dst_corners)
src_tf_im = warp(src_im, tf_model.inverse)

im0_,im1_ = _get_stitch_images(dst_im,src_im,tf_model=tf_model)
merged = _merge_stitch_images(im0_, im1_, mask_idx=1) #im1 is mask door maskidx

