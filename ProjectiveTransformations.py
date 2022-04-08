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


def _get_stitch_images(im0, im1, tf_model = None, n_keypoints=500,
                       min_samples=4, residual_threshold=2, **kwargs):
    """
    Alligns transformed source image to destination coördinates

    :param im0: src image
    :param im1: dst image
    :param tf_model: transformation model
    :param n_keypoints: amount of keypoints for calling new tf_model when tf_model=0
    :param min_samples: Ransac parameter,
        int in range (0, N)
        The minimum number of data points to fit a model to.
    :param residual_threshold: residual_threshold: Ransac parameter,
        float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    :return: transformed source image in the correct position according to destination image
    """
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
    new_shape = np.ceil(new_shape[::-1]).astype(int)    # groot genoege output shape

    shift = SimilarityTransform(translation = -min_corner)  # voor positieve coördinaten
    im0_ = warp(im0, shift.inverse, output_shape=new_shape, cval=-1)
    im1_ = warp(im1, (tf_model+shift).inverse, output_shape=new_shape, cval=-1)     # cval toont welke pixels achtergrond
    return im0_, im1_

def _merge_stitch_images(im0_,im1_,mask_idx=None,cval=0):
    """
    merges source image to destination coordinates

    :param im0_: alligned, transformed source image
    :param im1_: destination image
    :param mask_idx: index of image used as mask
    :param cval: set backgroundpixels of merged image to cval
    :return: image where source image is placed on destination image
    """
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

def _stitch(im0 ,im1 , mask_idx = None, cval = 0, show = True,tf_model=None,
                 n_keypoints=500, min_samples=4, residual_threshold=2, **kwargs):
    """
    combine _get_stitch_images and _merge_stitch_images

    :param im0: src image
    :param im1: dst image
    :param tf_model: transformation model
    :param n_keypoints: amount of keypoints for calling new tf_model when tf_model=0
    :param min_samples: Ransac parameter,
        int in range (0, N)
        The minimum number of data points to fit a model to.
    :param residual_threshold: residual_threshold: Ransac parameter,
        float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    :param mask_idx: index of image used as mask
    :param cval: set backgroundpixels of merged image to cval
    :param show: show the images produced by _get_stitch_images
    :return: image where source image is placed on destination image
    """
    im0_, im1_ = _get_stitch_images(im1, im0, tf_model=tf_model, n_keypoints= n_keypoints, min_samples=min_samples,
                                    residual_threshold= residual_threshold, **kwargs)
    if show:
        Plots = []
        Titles = []
        Plots.append(im0_); Titles.append("destination image")
        Plots.append(im1_); Titles.append("transformed source image")
        plot_figures('images', np.array(Plots), Titles, rowSize=2)
    merged = _merge_stitch_images(im0_, im1_, mask_idx, cval, **kwargs)
    return merged

if __name__ == "__main__":
    src_im = iio.imread('imgs/daenerys.jpg')
    dst_im = iio.imread('imgs/times-square.jpg')

    Plots = []
    Titles = []

    r, c = src_im.shape[:2]
    src_corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])    #xy formaat

    plt.figure(); plt.imshow(dst_im); plt.suptitle('input-> linksboven, linksonder, rechtboven, rechtsonder', fontsize=15)
    dst_corners = np.array(plt.ginput(4, 0))    # corners te vervangen oppervlak input-> linksboven, linksonder, rechtboven, rechtsonder
    print(dst_corners)

    tf_model = ProjectiveTransform()
    tf_model.estimate(src_corners, dst_corners)
    src_tf_im = warp(src_im, tf_model.inverse)  #transformeert foto naar juiste vorm

    Plots = []
    Titles = []

    merged = _stitch(src_im,dst_im, mask_idx=1, tf_model=tf_model)
    Plots.append(merged); Titles.append("Merged image")
    plot_figures('images', np.array(Plots), Titles, rowSize=1)

