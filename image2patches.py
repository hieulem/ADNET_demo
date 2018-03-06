
from models.models import create_model
from data.png_dataset import PngDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
from data.data_loader import CreateDataLoader
import matplotlib
matplotlib.use('agg')

import torch
import os.path
import argparse
from scipy import misc
import matplotlib.pyplot as plt
from m_util import sdmkdir

opt = argparse.ArgumentParser().parse_args()
opt.imfold='/gpfs/projects/LynchGroup/Train_all/CROPPED/p1000/'
opt.patch_fold_A = opt.imfold+'PATCHES/A/'
opt.patch_fold_B = opt.imfold+'PATCHES/B/'
A_fold = opt.imfold + 'A/'
B_fold = opt.imfold +  'B/'
opt.step = 64
opt.size = 256
imlist=[]
imnamelist=[]
sdmkdir(opt.patch_fold_A)
sdmkdir(opt.patch_fold_B)
for root,_,fnames in sorted(os.walk(A_fold)):
    for fname in fnames:
        if fname.endswith('.png'):
            path = os.path.join(root,fname)
            path_mask = os.path.join(B_fold,fname)
            imlist.append((path,path_mask,fname))
            imnamelist.append(fname)
print(imlist)

def savepatch_test(png,w,h,step,size,basename):

    ni = np.int32(np.floor((w- size)/step) +2)
    nj = np.int32(np.floor((h- size)/step) +2)

    for i in range(0,ni-1):
        for j in range(0,nj-1):
            misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(basename+format(i,'03d')+'_'+format(j,'03d')+'.png')
    for i in range(0,ni-1):
#        patches[i,nj-1,:,:,:] = png[:,i*step:i*step+size,h-size:h]
        misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(basename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


    for j in range(0,nj-1):
#        patches[ni-1,j,:,:,:] = png[:,w-size:w,j*step:j*step+size]
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(basename+format(ni-1,'03d')+'_'+format(j,'-3d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(basename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')

opt.no_dropout = True
model = create_model(opt)
print(model)
#for im_path,mask_path,imname in  imlist:
#    png = misc.imread(im_path,mode='RGB')
#    mask = misc.imread(mask_path,mode='L')
#    w,h,z = png.shape
    
#    savepatch_test(png,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname)
     
for root,_,fnames in sorted(os.walk(opt.patch_fold_A)):
    for fname in fnames:
        if fname.endswith('png'):
            im = misc.imread(os.path.join(root,fname))
            imtensor = torch.from_numpy(im).float().div(255)
            print(imtensor)
            temp = model.get_prediction(imtensor)
                
