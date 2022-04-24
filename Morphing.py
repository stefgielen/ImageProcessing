import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
import skimage
from skimage import img_as_ubyte, io, color
from scipy.spatial import Delaunay
from skimage.transform import AffineTransform, warp,resize
from skimage.draw import polygon, rectangle_perimeter
import cv2
from Piecewise_warp import warp_image 

def add_corners(pts, image):
    """
    adds corner and edge points
    :param pts: list of facial recognition points
    :param image: image
    :return: pts with corner points and edge points
    """
    # werkt enkel bij figuur met 1 face
    h, w, z = image.shape
    points = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, 0], [0, h / 2], [w, h / 2], [w / 2, h]])
    pts[0] = np.vstack((points, pts[0]))
    return pts


def get_points(image):
    """
    Uses the shape predictor 68 face landmarks to find the face recognition points in an image
    :param image:image
    :return: list of array with points (for each face in the image)
    """
    gray = img_as_ubyte(color.rgb2gray(image))
    rects = detector(gray, upsample_num_times=0)
    pts = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        pts.append(shape)

    pts = add_corners(pts, image)
    pts = pts[0]
    return pts


def get_triangular_mesh_t(image, pts):
    """
    :param image: image (only used when plotting the result)
    :param pts: points determined by the face landmarks predictor
    :return: points of triangles that devide the image in pieces
    """
    tris = Delaunay(pts)
   # fig, (ax) = plt.subplots(nrows=1, ncols=1)
   # ax.imshow(image)
   # ax.triplot(pts[:, 0], pts[:, 1], tris.simplices)
   # plt.title("triangular mesh")
   # plt.show()
    return tris

def face_morph1(img1,img2,alpha): #,lmarks=False,dtriang=False
    """
    deze functie gaat voor elke alpha waarde (1 per 1) morphen
    img1: image waarvan morphing start
    img2: image waar morphing naar overgaat en eindigt
    """
    
    img1 = io.imread(img1)
    img2=io.imread(img2)
    
    img1=img1[:,:,::-1] # in rgb 
    img2=img2[:,:,::-1]
 
    #resize
    img2=resize(img2,img1.shape[:2])
    
    pts1 = get_points(img1)
    pts2 = get_points(img2)

    tris1 = get_triangular_mesh_t(img1, pts1)
    tris2 = get_triangular_mesh_t(img2, pts2)

    ptsm = (1 - alpha) * pts1 + alpha * pts2

    warped1 = warp_image(img1,pts1,tris1,ptsm,img2.shape)
    warped2 = warp_image(img2,pts2,tris2,ptsm,img1.shape)
    morphed = img_as_ubyte((1-alpha) * warped1 + alpha * warped2)
    
   # plt.imshow(morphed)
   # plt.show()
    return morphed

def face_morph2(img1,img2,alphas):
    """
    Deze functie zal over elke alpha waarde lopen via een for lus en dan voor elke alpha waarde 
    gebruik maken van functie face_morph1. Elke alpha waarde heeft zijn bijhorend beeld die worden opgeslaan de array
    tot_frames en die dan op het einde gereturned wordt. Die array met al die beelden wordt dan
    via de methode videowriter tot een video omgezet.
    img1: image waarvan morphing start
    img2: image waar morphing naar overgaat en eindigt
    alphas: array van alpha waarden met een opgegeven stapgrootte (zie linespace)
    """
    tot_frames=[]
    for i in alphas:
        frame=face_morph1(img1,img2,i)
        print(i)
        tot_frames.append(frame)
    return tot_frames

def save_frames_to_video(file_path,frames,fps):  
    """
    file_path: plaats waar video wordt opgeslaan
    frames: de gereturnde array met alle beelden met respectievelijke alpha waarden van methode face_morph2
    fps:aantal frames per second
    """
    w,h,r=np.shape(frames[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # codec voor  frames te comprimeren
    writer=cv2.VideoWriter(file_path,fourcc,fps,(h,w))
    for i in frames:
        writer.write(i)

    writer.release() 
    
    
if __name__ == "__main__":
    
    img1 = './imgs/faces/daenerys.jpg'    #'./imgs/jeffmabilde.jpg'
    img2 ='./imgs/faces/gal_gadot.jpg'    #'./imgs/maris.jpg' #./imgs/stefmetzonnebril.jpg'
    p = "./shape_predictor_68_face_landmarks.dat"  #shape_predictor_68_face_landmarks_GTX
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    frames =face_morph2(img1,img2,alphas=np.linspace(0,1,30))
    save_frames_to_video('./imgs/my_morph_video_test.mp4',frames,30)