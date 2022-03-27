#!/usr/bin/env python
# coding: utf-8

# In[29]:


#custom restoration filters
import scipy
import skimage
import imageio as io
import matplotlib.pyplot as plt
from skimage import restoration, util
from dftfilt import dftfilt
from lpcfilter import lpcfilter
#from my_module import dftfilt,lpcfilter

def get_inv_H(H,cutoff=None, eps=1e-16):
    Hc= np.clip(H,eps,None)
    Hi=1/Hc
    if cutoff is not None:
        r,c=H.shape
        R,C=np.ogrid[:r,:c]
        D=np.sqrt((R-r//2)**2+(C-c//2)**2)
        Hi[D>cutoff]=eps
    return Hi

def get_wiener_H(H,k=1,Sn=None,Sf=None):
    #doel: mean square error the minimaliseren e²
    #k=Sn/Sf
    '''  
      if Sn is not None and Sf is not None:
        k=Sn/Sf
    else :
        k= 0.1  #trial&error en dan waarde pakken die voor beste image zorgt (Sn/Sf)
    '''
    Hw=(1/H)*np.abs(H**2)/(np.abs(H**2)+k)
    return Hw
    
def get_geomean_H(H,alpha=1,beta=1,Sn=None,Sf=None):
    Hg=((np.conj(H)/np.abs(H**2))**alpha)*(np.conj(H)/(np.abs(H**2)+beta*(Sn/Sf)))**(1-alpha)
    return Hg




# In[30]:


img = io.imread('imgs/obelix.tif')
img = util.img_as_float(img)
    #filter zonder padding
r, c = img.shape
H = lpcfilter((r, c), ftype='gaussian', D0=30)
g = dftfilt(img, H)
    #filter met padding
Hp = lpcfilter((2 * r, 2 * c), ftype='gaussian', D0=2 * 30)
gp = dftfilt(img, Hp, pad=True)

gn =util.random_noise(g,mode='s&p',amount=0.001)
gnm=scipy.ndimage.median_filter(gn,3)

#Inverting
fn_ei=dftfilt(gn,1/H)
fnm_ei=dftfilt(gnm,1/H)
cutoffs = [30,70,100]
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(8,3))
for i,x in enumerate(cutoffs):
    Hi=get_inv_H(H,cutoff=x)
    f_est=dftfilt(gnm,Hi)
    axes[i].imshow(f_est,cmap='gray');
    axes[i].axis('off');
    axes[i].set_title('cutoff={:1.0f}'.format(x))
    
#Wiener    
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,8))
Hw=get_wiener_H(H,k=0.01)
fn_ew=dftfilt(gn,Hw)
fnm_ew=dftfilt(gnm,Hw)
#Wiener_Restored_noisy_image
axes[0].imshow(fn_ew,cmap='gray')
axes[0].axis('off');
axes[0].set_title('Restored noisy image via Wiener'.format(x))
#Wiener_Restored_denoised_image
axes[1].imshow(fnm_ew,cmap='gray')
axes[1].axis('off');
axes[1].set_title('Restored denoised image via Wiener'.format(x))

#Geometric mean filtering
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,8))
Hg=get_geomean_H(H,alpha=0.5,beta=1,Sn=0.1,Sf=1)  # alpha=1/2 zorgt voor geometric mean filtering
fn_eg=dftfilt(gn,Hg)
fnm_eg=dftfilt(gnm,Hg)
#GMF_Restored_noisy_image
axes[0].imshow(fn_eg,cmap='gray')
axes[0].axis('off');
axes[0].set_title('Restored noisy image  via GMF'.format(x))
#GMF_Restored_denoised_image
axes[1].imshow(fnm_eg,cmap='gray')
axes[1].axis('off');
axes[1].set_title('Restored denoised image via GMF'.format(x))

