from matplotlib import pyplot as plt
import imageio as io
import skimage.transform
import numpy as np


def get_gaussian_pyramid(image,downscale=2, **kwargs):
    """
    get the gaussian pyramid of an image
    :param image: input image
    :param downscale: downscaling rate
    :param kwargs:
    :return: list of gaussian pyramid levels
    """
    images = []
    images.append(image)
    while images[-1].shape[0]%2==0 &  images[-1].shape[1]%2==0:
        images.append(skimage.transform.pyramid_reduce(images[-1], downscale=downscale, channel_axis=2, preserve_range=True))
    return images


def get_laplacian_pyramid(img_pyr, upscale=2, **kwargs):
    """
    get laplacian pyramid from an image's gaussian pyramid or the image itself
    :param img_pyr: gaussian pyramid of image
    :param upscale: upscale factor
    :param kwargs:
    :return: list of laplacian pyramid levels
    """
    images = []
    if not isinstance(img_pyr, list):           #als input image is en geen pyramid -> pyramid aanmaken (zie functiebeschrijving)
        img_pyr = get_gaussian_pyramid(img_pyr)
    for i in range(1, len(img_pyr)):
        images.append(img_pyr[i-1]-(skimage.transform.pyramid_expand(img_pyr[i]/255, upscale, channel_axis=2, preserve_range=True)*255))
    #first laplacian is the same as the highest level gaussian
    images.append(img_pyr[-1])
    return images


def plot_pyramid(pyramid):
    """
    create and plot titched image of a pyramid
    :param pyramid: list of laplacian or gaussian pyramid levels
    """
    pyramid_image = np.zeros([len(pyramid[0]), len(pyramid[0][0])+len(pyramid[1][0]),3]).astype(int)
    pyramid_image[:len(pyramid[0]), :len(pyramid[0][0])] = pyramid[0]
    xborder = 0
    for pic in pyramid[1:]:
        pyramid_image[xborder:xborder+len(pic), len(pyramid[0][0]):len(pyramid[0][0])+len(pic[0])] = pic
        xborder += len(pic)
    plt.imshow(pyramid_image)
    plt.show()


def reconstruct_image_from_laplacian_pyramid(pyramid):
    """
    create and plot a stitched image of a pyramid
    :param pyramid: list of laplacian pyramid levels
    """
    reconstructed_image = pyramid[-1]
    for i in range(2, len(pyramid)+1):
        reconstructed_image = (pyramid[-i] + skimage.transform.pyramid_expand(reconstructed_image / 255, 2, channel_axis=2, preserve_range=True) * 255)
    plt.imshow(reconstructed_image/255)
    plt.show()

if __name__ == "__main__":
    image =io.imread('./imgs/faces/superman.jpg')
    gpyramid = get_gaussian_pyramid(image)
    lpyramid = get_laplacian_pyramid(gpyramid)
    plot_pyramid(gpyramid)
    plot_pyramid(lpyramid)
    reconstruct_image_from_laplacian_pyramid(lpyramid)
