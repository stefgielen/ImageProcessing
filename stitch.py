import imageio
from skimage.transform import rescale
from matplotlib import pyplot as plt
from ProjectiveTransformations import _stitch

files = ['DFM_4209.jpg', 'DFM_4210.jpg','DFM_4211.jpg']
ims = []
for i,file in enumerate(files):
    im = imageio.imread('./imgs/' + file)
    im = im[:, 500:500+1987, :]
    ims.append(rescale(im,0.25,anti_aliasing=True,multichannel = True))

def stitch(ims, order=[1,0,2],mask_idx=None,tf_model='auto',
           n_keypoints=1000,min_samples=4,residual_trashold=2,**kwargs):
    ims_sorted = []
    for i in range (len(ims)):
        ims_sorted.append(ims[order[i]])
    cval = -1
    if(len(ims_sorted)== 2):
        cval = 0
    merged = _stitch(im0=ims_sorted[0], im1=ims_sorted[1], cval=cval, tf_model=tf_model)
    for i in range(2,len(ims_sorted)):
        cval = -1
        if(i==len(ims_sorted)-1):
            cval = 0
        merged = _stitch(im0=merged, im1=ims_sorted[i], cval=cval, tf_model=tf_model)
    return merged

if __name__ == "__main__":
    merged = stitch(ims)
    plt.imshow(merged)
    plt.show()