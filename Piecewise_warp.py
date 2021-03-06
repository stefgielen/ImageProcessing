import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
import skimage
from skimage import img_as_ubyte, io, color
from scipy.spatial import Delaunay
from skimage.transform import AffineTransform, warp, resize
from skimage.draw import polygon, rectangle_perimeter


def add_corners(pts, image):
    """
    adds corner and edge points
    :param pts: list of facial recognition points
    :param image: image
    :return: pts with corner points and edge points
    """
    h, w, z = image.shape
    points = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [0, h / 2], [w, h / 2], [w / 2, h]])
    #points = np.array([[1, 1], [w-1, 1], [1, h-1], [w-1, h-1], [w / 2, 1], [1, h / 2], [w-1, h / 2], [w / 2, h-1]])
    pts = np.vstack((points, pts))
    return pts


def get_points(image):
    """
    Uses the shape predictor 68 face landmarks to find the face recognition points in an image
    :param image:image
    :return: list of array with points (for each face in the image)
    """
    p = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    gray = img_as_ubyte(color.rgb2gray(image))
    rects = detector(gray, upsample_num_times=0)
    pts = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        pts.append(shape)
    return pts


def get_triangular_mesh_t(image, pts):
    """

    :param image: image (only used when plotting the result)
    :param pts: points determined by the face landmarks predictor
    :return: points of triangles that devide the image in pieces
    """
    tris = Delaunay(pts)

    """fig, (ax) = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image)
    ax.triplot(pts[:, 0], pts[:, 1], tris.simplices)
    plt.title("triangular mesh")
    plt.show()"""

    return tris


def get_max_shape(image1, image2_shape):
    """
    :return: the maximum width, heigth and the third index
    """
    w1, h1, z1 = image1.shape
    w2, h2, z2 = image2_shape
    w = max(w1, w2)
    h = max(h1, h2)
    return w, h, z1


def get_bounding_box(t1):
    """
    :param t1: three points that form a triangle on the image
    :return: rectangle that surrounds the triangle
    """
    start_x = min(t1[:, 0])
    start_y = min(t1[:, 1])
    w = max(t1[:, 0]) - min(t1[:, 0])
    h = max(t1[:, 1]) - min(t1[:, 1])
    array = (start_x, start_y, w, h)
    return array


def warp_triangle(image, bb, M, output_shape):
    """
    :param image: Image that is going to be warped
    :param bb: bounding box, selects section of image
    :param M: Transformation
    :param output_shape: bounding box of output shape
    :return: part of the image selected by the boundingbox transformed with the transformation M to the shape of output_shape
    """
    sub_image = image[int(bb[1]):int(bb[1] + bb[3]), int(bb[0]):int(bb[0] + bb[2])]
    wt = warp(sub_image, M, output_shape=output_shape)
    return wt


def get_triangle_mask(tm, bbm, output_shape):
    """
    :param tm: points of triangle
    :param bbm: points of bounding box surrounding the triangle
    :param output_shape:
    :return: Mask of zero's with the measurements of output_shape  and ones in the shape of the triangle
    """
    tm = tm.astype(int)
    rr, cc = polygon(tm[:, 1] - int(bbm[1]), tm[:, 0] - int(bbm[0]), output_shape)
    mask = np.zeros([output_shape[0], output_shape[1], 3])
    mask[rr, cc, :] = 1
    return mask


def warp_image(im1, pts, tris, ptsm, im2_shape):
    """
    :param im1: image that need warping
    :param pts: face landmark points
    :param tris: array of triangles dividing image 1 in pieces
    :param ptsm: points defined by intermediate key points
    :param im2_shape: shape of second image
    :return: warped image
    """
    max_shape_of_im1_and_im2 = get_max_shape(im1, im2_shape)
    warped = np.zeros(max_shape_of_im1_and_im2)

    for tri in tris.simplices:
        t1 = np.array([pts[tri[0]], pts[tri[1]], pts[tri[2]]])
        tm = np.array([ptsm[tri[0]], ptsm[tri[1]], ptsm[tri[2]]])
        bb1 = get_bounding_box(t1)
        bbm = get_bounding_box(tm)

        """
        #### print
        row, col = rectangle_perimeter(start=(bb1[1], bb1[0]), end=(bb1[1] + bb1[3], bb1[0] + bb1[2]))
        roww, coll = polygon(t1[:, 1], t1[:, 0])

        fig, (ax) = plt.subplots(nrows=1, ncols=1)
        ax.imshow(image1)
        ax.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
        ax.plot(col, row, "--y")
        ax.plot(coll, roww, 'g')
        plt.title("triangular mesh")
        plt.show()
        """
        ####

        M = AffineTransform()
        M.estimate(t1 - bb1[:2], tm - bbm[:2])

        if not np.linalg.det(M.params):
            continue
        else:
            M = np.linalg.inv(M.params)
            output_shape = warped[int(bbm[1]):int(bbm[1] + bbm[3]), int(bbm[0]):int(bbm[0] + bbm[2])].shape[:2]

            wt1 = warp_triangle(im1, bb1, M, output_shape)

            """plt.imshow(wt1)
            plt.title("warp_triangle")
            plt.show()"""

            mask = get_triangle_mask(tm, bbm, output_shape)
            warped[int(bbm[1]):int(bbm[1] + bbm[3]), int(bbm[0]):int(bbm[0] + bbm[2])] = warped[
                                                                                         int(bbm[1]):int(
                                                                                             bbm[1] + bbm[3]),
                                                                                         int(bbm[0]):int(
                                                                                             bbm[0] + bbm[2])] * (
                                                                                                 1 - mask) + mask * wt1
    return warped


# main
if __name__ == "__main__":
    image1 = io.imread('./imgs/faces/daenerys.jpg')
    image2 = io.imread('./imgs/faces/gal_gadot.jpg')
    # image1=resize(image1,image2.shape[:2])
    image2 = resize(image2, image1.shape[:2])

    pts1 = get_points(image1)
    pts1 = pts1[0]
    pts1 = add_corners(pts1, image1)

    plt.imshow(image1)
    plt.plot(pts1[:, 0], pts1[:, 1], 'o')
    plt.show()

    pts2 = get_points(image2)
    pts2 = pts2[0]
    pts2 = add_corners(pts2, image2)

    tris1 = get_triangular_mesh_t(image1, pts1)
    tris2 = get_triangular_mesh_t(image2, pts2)

    plt.imshow(image1)
    plt.triplot(pts1[:, 0], pts1[:, 1], tris1.simplices)
    plt.show()

    alpha = 0.5
    ptsm = (1 - alpha) * pts1 + alpha * pts2

    warped1 = warp_image(image1, pts1, tris1, ptsm, image2.shape)
    warped2 = warp_image(image2, pts2, tris2, ptsm, image1.shape)
    morphed = img_as_ubyte((1 - alpha) * warped1 + alpha * warped2)
    plt.imshow(morphed)
    plt.show()