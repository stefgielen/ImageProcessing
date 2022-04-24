from skimage import img_as_ubyte, io, color
from Piecewise_warp import get_points, get_triangular_mesh_t, warp_image, add_corners, get_triangle_mask
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from pyramid_blending import get_laplacian_pyramid, reconstruct_image_from_laplacian_pyramid, get_gaussian_pyramid, plot_pyramid
from Functions import plot_figures


def get_mask(hull, output_shape):

    mask = np.zeros([output_shape[0], output_shape[1], 3])
    rowpoints = []
    colpoints = []
    for vertex in hull.vertices:
        rowpoints.append(hull.points[vertex][0])
        colpoints.append(hull.points[vertex][1])

    rr, cc = polygon(colpoints, rowpoints, output_shape)
    mask[rr, cc,:] = 1

    return mask


def padcrop(img1, img2shape):
    """
    pad or crop image 1 to the size of image 2
    :param img1: image to be padded/cropped
    :param img2shape: shape to be padded/cropped to
    :return: cropped/padded image
    """

    if img1.shape[0] < img2shape[0]:   #img1 is kleiner -> padden tot img2shape[0]
        img1 = np.pad(img1, ((0, img2shape[0]-img1.shape[0]), (0, 0), (0, 0)), 'constant')
    else:                               #img1 is groter ->  cropppen tot img2shape[0]
        img1 = img1[:img2shape[0], :]
    if img1.shape[1] < img2shape[1]:   # img1 is kleiner -> padden tot img2shape[1]
        img1 = np.pad(img1, ((0, 0), (0, img2shape[1]-img1.shape[1]), (0, 0)), 'constant')
    else:                               # img1 is groter ->  cropppen tot img2shape[1]
        img1 = img1[:, :img2shape[1]]
    return img1


def pyramidblend(imgor, warped, mask):
    """
    perform pyramid blending and plot
    :param imgor: original image
    :param warped: warped image
    :param mask: mask of face in original image

    """
    imLapl = get_laplacian_pyramid(imgor)
    warpLapl = get_laplacian_pyramid(warped)
    maskGaus = get_gaussian_pyramid(mask)
    blendLapl = []  # np.zeros_like(im1Lapl)
    for i in range(0, len(imLapl)):
        blendLapl.append((imLapl[i] * (1 - maskGaus[i])) + (warpLapl[i] * maskGaus[i]))

    swapped = reconstruct_image_from_laplacian_pyramid(blendLapl)

    return swapped

def swap_faces(img1, img2=None, blendmode='pyramid', faceorder=(0, 1), display = True, flip_faces=(False,False)):
    """
    Swap face in img1 with face in img2 using the specified blend method.
    if img2==None then img1 should contain two faces and the faceorder argument specifies swapping order.
    blendmode:'alpha'(simple alpha-blending),
        'pyramid'(use pyramid_blend()),
        'cv2.seamlessClone'(use cv2.seamlessClone function)
    flip_faces:(True,True), True flips each replacement face(e.g. because both faces are orientated in
    different left-right directions in the same image).
    """
    plots = []
    titles = []

    # Step1: Get feature points
    if img2 is None:
        pts = get_points(img1)
        pts1 = pts[0]
        pts2 = pts[1]

    else:
        pts1 = get_points(img1)[0]
        pts2 = get_points(img2)[0]

    if display:
        if img2 is None:
            fig = plt.figure()
            plt.plot(pts1[:, 0], pts1[:, 1], 'o')
            plt.plot(pts2[:, 0], pts2[:, 1], 'o')
            plt.imshow(img1, cmap='gray')
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("img1 get_points")
            ax.plot(pts1[:, 0], pts1[:, 1], 'o')
            ax.imshow(img1, cmap='gray')

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("img2 get_points")
            ax.plot(pts2[:, 0], pts2[:, 1], 'o')
            ax.imshow(img2, cmap='gray')
            plt.show()

    # select convexhull
    hull1 = ConvexHull(pts1)
    hull2 = ConvexHull(pts2)

    if display:
        if img2 is None:
            plt.imshow(img1)
            plt.plot(pts1[:, 0], pts1[:, 1], 'o')
            plt.plot(pts2[:, 0], pts2[:, 1], 'o')

            for simplex in hull1.simplices:
                plt.plot(pts1[simplex, 0], pts1[simplex, 1], 'k-')
                plt.plot(pts2[simplex, 0], pts2[simplex, 1], 'k-')
            plt.title("convex hull")
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("img1 convex_hull")
            ax.plot(pts1[:, 0], pts1[:, 1], 'o')
            for simplex in hull1.simplices:
                ax.plot(pts1[simplex, 0], pts1[simplex, 1], 'k-')
            ax.imshow(img1, cmap='gray')

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("img2 convex_hull")
            ax.plot(pts2[:, 0], pts2[:, 1], 'o')
            for simplex in hull2.simplices:
                ax.plot(pts2[simplex, 0], pts2[simplex, 1], 'k-')
            ax.imshow(img2, cmap='gray')
            plt.show()

    # step 2: add corners
    if img2 is None:
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img1)

    else:
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img2)

    # Step 3:Get Delaunay triangulation
    if img2 is None:
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img1, pts2)
    else:
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img2, pts2)

    if display:
        if img2 is None:
            plt.imshow(img1)
            plt.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
            plt.triplot(pts2[:, 0], pts2[:, 1], tris2.simplices)
            plt.title("triangular mesh ")
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("img1 triangular mesh")
            ax.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
            ax.imshow(img1, cmap='gray')

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title("img2 triangular mesh")
            ax.triplot(pts2[:, 0], pts2[:, 1], tris2.simplices)
            ax.imshow(img2, cmap='gray')
            plt.show()

    # Step 4: warp image
    alpha = 0
    ptsm_alpha0 = (1 - alpha) * pts1 + alpha * pts2
    alpha=1
    ptsm_alpha1 =(1 - alpha) * pts1 + alpha * pts2
    if img2 is None:
        warped1 = warp_image(img1, pts1, tris1, ptsm_alpha1, img1.shape)*255
        warped2 = warp_image(img1, pts2, tris2, ptsm_alpha0, img1.shape)*255
    else:
        warped1 = warp_image(img1, pts1, tris1, ptsm_alpha1, img2.shape)*255
        warped2 = warp_image(img2, pts2, tris2, ptsm_alpha0, img1.shape)*255

    # Step 5: Create masks
    mask1 = get_mask(hull1, warped1.shape[:2])
    mask2 = get_mask(hull2, warped2.shape[:2])

    if display:
        if img2 is None:
            img2 = np.copy(img1)
        plots = [img1, mask1, warped2/255, img2, mask2, warped1/255]
        titles = ["img1", "img1 mask", "img2 warped", "img2", "img2 mask", "img1 warped"]
        fig = plt.figure()
        for i in range(len(plots)):
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_title(titles[i])
            ax.imshow(plots[i], cmap='gray')
        plt.show()

    # Step 6: Pad or crop to same size as image 1
    if img2 is not None:
        mask1 = padcrop(mask1, img1.shape)      #mask van gezicht image 1
        warped1 = padcrop(warped1, img2.shape)  #gewarpete gezicht image 2 in vorm image 1
        mask2 = padcrop(mask2, img2.shape)
        warped2 = padcrop(warped2, img1.shape)

    if display:
        if img2 is None:
            img2 = np.copy(img1)
        plots = [img1, mask1, warped2/255, img2, mask2, warped1/255]
        titles = ["img1", "img1 mask", "img2 warped", "img2", "img2 mask", "img1 warped"]
        fig = plt.figure()
        for i in range(len(plots)):
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_title(titles[i])
            ax.imshow(plots[i], cmap='gray')
        plt.show()

    # Step 7: Blend images using mask

    if (img2 is None):
        if blendmode == 'alfa-blending':
            swapped = ((warped1 * mask2) + (warped2*mask1) + img1 * (1 - mask1)*(1-mask2))/255
            plt.imshow(swapped)
            plt.show()
        elif blendmode == 'pyramid':
            tempswapped = pyramidblend(img1, warped1, mask2)
            im1Lapl = get_laplacian_pyramid(tempswapped)
            warpLapl = get_laplacian_pyramid(warped2/255)
            maskGaus = get_gaussian_pyramid(mask1)
            blendLapl = []  # np.zeros_like(im1Lapl)
            for i in range(0, len(im1Lapl)):
                blendLapl.append((im1Lapl[i] * (1 - maskGaus[i])) + (warpLapl[i] * maskGaus[i]))

            swapped = reconstruct_image_from_laplacian_pyramid(blendLapl)*255
            plt.imshow(swapped)
            plt.show()

    else:
        if blendmode == 'alfa-blending':
            swapped1 = (warped2 * mask1) + img1 * (1 - mask1)
            swapped2 = (warped1 * mask2) + img2 * (1 - mask2)
            plt.imshow(swapped1/255)
            plt.show()
            plt.imshow(swapped2/255)
            plt.show()

        if blendmode == 'pyramid':
            swapped1 = pyramidblend(img1, warped2, mask1)
            swapped2 = pyramidblend(img2, warped1, mask2)
            plt.imshow(swapped1)
            plt.show()
            plt.imshow(swapped2)
            plt.show()
# main
if __name__ == "__main__":
    image1 = io.imread('./imgs/faces/superman.jpg')
    image2 = io.imread('./imgs/faces/nicolas_cage.jpg')
    image3 = io.imread('./imgs/faces/brangelina.jpg')
    image4 = io.imread('/Users/stefgielen/Downloads/878873_1_seoimage4x3_bn-859569_e58c83605ff84e42832fce5b82595756.jpg')
    image5 = io.imread('/Users/stefgielen/Downloads/https---static.nieuwsblad.be-Assets-Images_Upload-2020-08-11-dc59c00e-dbf1-11ea-8ffb-9e26f69cae24.jpg')


    swap_faces(image4, image5, blendmode="pyramid", flip_faces=(False,True))
    swap_faces(image1, image2, blendmode="pyramid")
    swap_faces(image3, None, blendmode='alfa-blending')
    swap_faces(image3, None, blendmode='pyramid')
