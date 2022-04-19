from skimage import img_as_ubyte, io, color
from Piecewise_warp import get_points, get_triangular_mesh_t, warp_image, add_corners, get_triangle_mask
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

def get_mask(tm ,output_shape):
    tm = tm.astype(int)
    rr, cc = polygon(tm[:, 1], tm[:, 0])
    mask = np.zeros([output_shape[0],output_shape[1],3])
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

    # Step1: Get feature points
    if (img2 == None):
        pts = get_points(img1)
        pts1 = pts[0]
        pts2 = pts[1]
    else:
        pts1 = get_points(img1)[0]
        pts2 = get_points(img2)[0]
    """
    plt.imshow(img1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'o')
    plt.show()
    """
    # select convexhull
    hull1 = ConvexHull(pts1)
    hull2 = ConvexHull(pts2)
    """
    plt.imshow(img1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'o')
    for simplex in hull1.simplices:
        plt.plot(pts1[simplex, 0], pts1[simplex, 1], 'k-')
    plt.show()
    """
    # step 2: add corners
    if (img2 == None):
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img1)
    else:
        pts1 = add_corners(pts1, img1)
        pts2 = add_corners(pts2, img2)
    """
    plt.imshow(img1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'o')
    plt.show()
    """
    # Step 3:Get Delaunay triangulation
    if (img2 == None):
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img1, pts2)
    else:
        tris1 = get_triangular_mesh_t(img1, pts1)
        tris2 = get_triangular_mesh_t(img2, pts2)

    # Step 4: warp image and create mask
    alpha = 0.5
    ptsm = (1 - alpha) * pts1 + alpha * pts2
    if (img2 == None):
        warped1 = warp_image(img1, pts1, tris1, ptsm, img1.shape)
        warped2 = warp_image(img1, pts2, tris2, ptsm, img1.shape)
    else:
        warped1 = warp_image(img1, pts1, tris1, ptsm, img1.shape)
        warped2 = warp_image(img2, pts2, tris2, ptsm, img2.shape)
    """
    plt.imshow(warped1)
    plt.show()
    """
    # Create mask
    x=pts1[hull1.vertices.tolist(),0]
    y=pts1[hull1.vertices.tolist(),1]
    contour=np.stack((x, y), axis=-1)
    print(contour)
    if (img2 == None):
        mask1 = get_mask(contour, warped2.shape[:2])
        #mask2 = get_triangle_mask(hull2.simplices, img1.shape[2], img1.shape[2])
    else:
        mask1 = get_triangle_mask(hull1.simplices, img1.shape[2], img1.shape[2])
        #mask2 = get_triangle_mask(hull1.simplices, img2.shape[2], img2.shape[2])
    """
    plt.imshow(mask1)
    plt.show()
    """


# main
if __name__ == "__main__":
    image1 = io.imread('.\imgs\daenerys_face.jpg')
    image2 = io.imread('.\imgs\gal_gadot_face.jpg')
    image3 = io.imread('./imgs/faces/brangelina.jpg')
    swap_faces(image3)
