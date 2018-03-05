import numpy as np
import cv2
from scipy import misc
import os 
from util import sdmkdir,to_rgb3b

def show_heatmap_on_image(img,mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(cam*255)

def show_plainmask_on_image(im,mask):
    mask = np.float32(mask)
    mask = mask/np.max(mask)
    im = np.float32(im)
    if im.ndim ==2:
        im = to_rgb3b(im)
    im = im * 0.75
    im[:,:,1] = im[:,:,1] + mask*30
    im[im>255] = 255
    return im.astype(np.uint8)
def visdir(imdir,maskdir,visdir):
    sdmkdir(visdir)    
    imlist=[]
    imnamelist=[]
    print imdir
    for root,_,fnames in sorted(os.walk(imdir)):
        print root,fnames
        for fname in fnames:
            if fname.endswith('.png'):
                pathA = os.path.join(root,fname)
                pathB = os.path.join(maskdir,fname)
                imlist.append((pathA,pathB,fname))
                imnamelist.append(fname)
    print(imlist)
    for pathA,pathB,fname in imlist:
        print fname
        A = misc.imread(pathA)
        B = misc.imread(pathB)
        vim = show_plainmask_on_image(A,B)
        cv2.imwrite(os.path.join(visdir,fname),vim)
def visAB(root):
    
    A = root + '/A/'
    B = root + '/B/'
    vis = root + '/vis/'
    visdir(A,B,vis)

root = '/gpfs/projects/LynchGroup/Train_all/CROZtrain/CROPPED/p1000/' 
root ='/gpfs/projects/LynchGroup/Train_all/CROPPED/p1000/'
visAB(root)
