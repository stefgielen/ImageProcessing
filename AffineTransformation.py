import cv2
import matplotlib.pyplot as plt

from skimage.transform import warp, AffineTransform
from skimage.feature import (match_descriptors, ORB, plot_matches)
from skimage.measure import ransac

import imageio as iio
import numpy as np
from Functions import plot_figures
from skimage import color, io


"-------------------------Affine Transformation estimation-----------------------------"

"-------------------------Scikit Image-----------------------------"


def get_matches(im_or, im_tf, n_keypoints = 500,
                ax = None, title = 'Original vs transformed'):
    """
    Extract descriptors and keypoints from two images and return the matches
    :param im_or: original image
    :param im_tf: transformed image
    :param n_keypoints: amount of keypoints to be found
    :param ax: axes object for plotting the keypoints as subplot
    :param title: title of the axes subplot
    :return:
        matches: Indices of corresponding matches in first and second set of
        descriptors
        keypoints_or:  Keypoint coordinates as ``(row, col)`` of original image
        keypoints_tf: Keypoint coordinates as ``(row, col)`` of transformed image
    """
    descriptor_extractor = ORB(n_keypoints = n_keypoints)   # descriptor extractor object ORB is extractor
    descriptor_extractor.detect_and_extract(color.rgb2gray(im_or))
    keypoints_or = descriptor_extractor.keypoints           # originele keypoints
    descriptors_or = descriptor_extractor.descriptors
    descriptor_extractor.detect_and_extract(color.rgb2gray(im_tf))
    keypoints_tf = descriptor_extractor.keypoints
    descriptors_tf = descriptor_extractor.descriptors
    matches = match_descriptors(descriptors_or, descriptors_tf, cross_check=True)

    if ax is not None:
        plot_matches(ax, im_or, im_tf, keypoints_or, keypoints_tf, matches)
        ax.axis('off')
        ax.set_title(title)
    return matches, keypoints_or, keypoints_tf


def get_tf_model(src, dst, xTransform=AffineTransform, n_keypoints=500,
                 min_samples=4, residual_threshold=2, **kwargs):
    """
    calls the ransac function that fits a model to data using the ransac algorithm
    :param src:  source image
    :param dst: destination image
    :param xTransform: Ransac parameter,
        model class, defines transformation type
    :param n_keypoints: get_matches parameter,
        int in range (0, N)
        Defines amount of detected matches
    :param min_samples: Ransac parameter,
        int in range (0, N)
        The minimum number of data points to fit a model to.
    :param residual_threshold: Ransac parameter,
        float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    :param kwargs:
    :return: returns tf_model alculated using RANSAC
    """
    matches, kp_src, kp_dst = get_matches(src, dst, n_keypoints= n_keypoints)
    src = kp_src[matches[:, 0]][:, ::-1]
    dst = kp_dst[matches[:, 1]][:, ::-1]
    tf_model, __ = ransac((src,dst), xTransform, min_samples=min_samples,
                          residual_threshold=residual_threshold, **kwargs)
    return tf_model

"-------------------------Affine Transformation-----------------------------"
if __name__ == "__main__":
    im = iio.imread('imgs/yoda.jpg')
        # Translatie
    c = np.array(im.shape[:2])//2   # center
    T = np.diag([1, 1, 1])          # diagonaalmatrix
    T[:2, -1] = -c[::-1]            # spaciale coördinaten
    imT = warp(im, np.linalg.inv(T), order=3)
        # Rotatie
    theta = np.deg2rad(30)          # radialen zetten
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])           # rotatiematrix R
    imR = warp(im, np.linalg.inv(R), order=3)
        # Translatie naar centrum
    Ti = np.diag([1, 1, 1])           # translatiematrix om naar centrum te schuiven
    Ti[:2, -1] = c[::-1]
    A = np.dot(Ti, np.dot(R, T))
    imA = warp(im, np.linalg.inv(A), order=3)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
    matches, kp_or, kp_tf = get_matches(im, imA, n_keypoints=200, ax=ax1)
    plt.show()

    tf_model = get_tf_model(im, imA, xTransform=AffineTransform, n_keypoints=200,
                            min_samples=4, residual_threshold=2)
    print("skimage: \n", tf_model.params, "\n \n", A)  # voorspeld


    "-------------------------Opencv-----------------------------"
    im_or, im_tf = im, (imA*255).astype(np.uint8)
    orb = cv2.ORB_create()
    kp_or, des_or = orb.detectAndCompute(im_or, None)
    kp_tf, des_tf = orb.detectAndCompute(im_tf, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_or, des_tf)
    matches = sorted(matches, key = lambda x:x.distance)
    im_or_tf = cv2.drawMatches(im_or, kp_or, im_tf, kp_tf, matches[:10], None, flags=2)

    plt.imshow(im_or_tf), plt.show()

    src_pts = np.float32([kp_or[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_tf[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    MA, mask = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 10.0)
    MP, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    print("openCV: \n", MA, "\n \n", MP)  # voorspeld

