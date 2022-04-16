from matplotlib import pyplot as plt
import imageio as io
import skimage.transform
import numpy as np

def get_gaussian_pyramid(image,downscale=2, **kwargs):
    images = []
    images.append(image)
    for i in range(1,6):
        images.append(skimage.transform.pyramid_reduce(image,downscale**i,channel_axis = 2,preserve_range=True))
    return images

def get_laplacian_pyramid(img_pyr,upscale=2, **kwargs):
    images = []
    for i in range(1,len(img_pyr)):
        images.append(img_pyr[i-1]-(skimage.transform.pyramid_expand(img_pyr[i]/255,upscale,channel_axis = 2,preserve_range=True)*255)+50)
    #first laplacian is the same as the highest level gaussian
    images.append(img_pyr[-1])
    return images

def plot_pyramid(pyramid):
    pyramid_image = np.zeros([len(pyramid[0]),len(pyramid[0][0])+len(pyramid[1][0]),3]).astype(int)
    pyramid_image[:len(pyramid[0]), :len(pyramid[0][0])] = pyramid[0]
    xborder = 0
    for pic in pyramid[1:]:
        pyramid_image[xborder:xborder+len(pic), len(pyramid[0][0]):len(pyramid[0][0])+len(pic[0])] = pic
        xborder += len(pic)
    plt.imshow(pyramid_image)
    plt.show()

if __name__ == "__main__":
    image =io.imread('./imgs/faces/superman.jpg')
    gpyramid = get_gaussian_pyramid(image)
    lpyramid = get_laplacian_pyramid(gpyramid)
    plot_pyramid(gpyramid)
    plot_pyramid(lpyramid)