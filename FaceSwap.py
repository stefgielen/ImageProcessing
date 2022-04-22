from skimage import img_as_ubyte, io, color
from Piecewise_warp import get_points, get_triangular_mesh_t, warp_image, add_corners, get_triangle_mask
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from pyramid_blending import get_laplacian_pyramid, reconstruct_image_from_laplacian_pyramid, get_gaussian_pyramid
from Functions import plot_figures

def get_mask(hull ,output_shape):


    mask = np.zeros([output_shape[0], output_shape[1], 3])
    rowpoints = []
    colpoints = []
    for vertex in hull.vertices:
        rowpoints.append(hull.points[vertex][0])
        colpoints.append(hull.points[vertex][1])

    rr, cc = polygon(colpoints, rowpoints, output_shape)
    mask[rr, cc,:] = 1

    return mask

def swap_faces(img1, img2=None, blendmode='pyramid', faceorder=(0, 1)):
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
    plots2=[]
    titles2=[]

    # Step1: Get feature points
    if img2 is None:
        pts = get_points(img1)
        pts1 = pts[0]
        pts2 = pts[1]

    else:
        pts1 = get_points(img1)[0]
        pts2 = get_points(img2)[0]

    """if img2 is None:
        plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        plt.title("feature points")
        plt.show()
    else:
        plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        # plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        plt.title("feature points")
        plt.show()
        plt.imshow(img2)
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        # plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        plt.title("feature points")
        plt.show()"""

    # select convexhull
    hull1 = ConvexHull(pts1)
    hull2 = ConvexHull(pts2)


    """if img2 is None:
        plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')

        for simplex in hull1.simplices:
            plt.plot(pts1[simplex, 0], pts1[simplex, 1], 'k-')
            plt.plot(pts2[simplex, 0], pts2[simplex, 1], 'k-')
        plt.title("convex hull")
        plt.show()
    else:
        plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        for simplex in hull1.simplices:
            plt.plot(pts1[simplex, 0], pts1[simplex, 1], 'k-')
        plt.title("convex hull")
        plt.show()

        plt.imshow(img2)
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        for simplex in hull1.simplices:
            plt.plot(pts2[simplex, 0], pts2[simplex, 1], 'k-')
        plt.title("convex hull")
        plt.show()"""

    # step 2: add corners
    if img2 is None:
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img1)

        """plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        plt.title("corner points")
        plt.show()"""
    else:
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img2)

        """plt.imshow(img1)
        plt.plot(pts1[:, 0], pts1[:, 1], 'o')
        plt.title("corner points")
        plt.show()

        plt.imshow(img2)
        plt.plot(pts2[:, 0], pts2[:, 1], 'o')
        plt.title("corner points")
        plt.show()"""


    # Step 3:Get Delaunay triangulation
    if img2 is None:
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img1, pts2)
    else:
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img2, pts2)

    """if img2 is None:
        plt.imshow(img1)
        plt.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
        plt.triplot(pts2[:, 0], pts2[:, 1], tris2.simplices)
        plt.title("triangular mesh ")
        plt.show()
    else:
        plt.imshow(img1)
        plt.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
        plt.title("triangular mesh ")
        plt.show()
        plt.imshow(img2)
        plt.triplot(pts2[:, 0], pts2[:, 1], tris2.simplices)
        plt.title("triangular mesh ")
        plt.show()"""

    # Step 4: warp image
    alpha = 0
    ptsm_alpha0  = (1 - alpha) * pts1 + alpha * pts2
    alpha=1
    ptsm_alpha1 =(1 - alpha) * pts1 + alpha * pts2
    if img2 is None:
        warped1 = warp_image(img1, pts1, tris1, ptsm_alpha1, img1.shape)
        warped2 = warp_image(img1, pts2, tris2, ptsm_alpha0, img1.shape)
    else:
        warped1 = warp_image(img1, pts1, tris1, ptsm_alpha1, img2.shape)
        warped2 = warp_image(img2, pts2, tris2, ptsm_alpha0, img1.shape)

    # Step 5: Create masks
    mask1 = get_mask(hull1, warped2.shape[:2])
    mask2 = get_mask(hull2, warped1.shape[:2])

    plots.append(img1)
    plots.append(mask1)
    plots.append(warped2)
    titles.append("")
    titles.append("")
    titles.append("")

    plots2.append(img2)
    plots2.append(mask2)
    plots2.append(warped1)
    titles2.append("")
    titles2.append("")
    titles2.append("")


    # Step 6: Pad or crop to same size as image 1
    if img2 is not None:
        if warped1.shape[0]< img1.shape[0]:
            mask1 = np.pad(mask1, ((mask1.shape[0], img1.shape[0]), 'constant'))
            warped1 = np.pad(warped1, ((warped1.shape[0], img1.shape[0]), 'constant'))
        else:
            mask1 = mask1[:img1.shape[0], :]
            warped1 = warped2[:img1.shape[0], :]
        if warped2.shape[1] < img1.shape[1]:
            mask1 = np.pad(mask1, ((mask1.shape[1], img1.shape[1]), 'constant'))
            warped1 = np.pad(warped1, ((warped1.shape[1], img1.shape[1]), 'constant'))
        else:
            mask1 = mask1[:, :img1.shape[1]]
            warped1 = warped1[:, :img1.shape[1]]

    plots.append(img1)
    plots.append(mask1)
    plots.append(warped2)
    titles.append("")
    titles.append("")
    titles.append("")
    plot_figures('face swap', np.array(plots), titles, rowSize=3)

    plots2.append(img2)
    plots2.append(mask2)
    plots2.append(warped1)
    titles2.append("")
    titles2.append("")
    titles2.append("")
    plot_figures('face swap', np.array(plots2), titles2, rowSize=3)

    # Step 7: Blend images using mask
    if blendmode == 'alfa-blending':
        swapped = (warped1 * mask1) + img1/255

    if blendmode == 'pyramid':
        im1Lapl = get_laplacian_pyramid(img1)
        warpLapl = get_laplacian_pyramid(warped1)
        maskGaus = get_gaussian_pyramid(mask1)
        blendLapl = np.zeros_like(im1Lapl)
        for i in range(0, len(im1Lapl)):
            blendLapl[i] = (im1Lapl[i] * (1-maskGaus[i])) + (warpLapl[i] * maskGaus[i])

        #blendLapl = im1Lapl * (1-maskGaus) + warpLapl * (maskGaus)
        swapped = reconstruct_image_from_laplacian_pyramid(blendLapl)

    plt.imshow(swapped)
    plt.show()
# main
if __name__ == "__main__":
    image1 = io.imread('./imgs/faces/superman.jpg')
    image2 = io.imread('./imgs/faces/nicolas_cage.jpg')
    image3 = io.imread('./imgs/faces/brangelina.jpg')
    swap_faces(image1, image2, blendmode="pyramid")