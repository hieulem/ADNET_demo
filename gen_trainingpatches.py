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
from m_util import sdmkdir,savepatch_test,savepatch_train,patches2png
from vis import visAB,visABC
opt = argparse.ArgumentParser().parse_args()
opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p300_3/'#+'/nfs/bigbox/hieule/p1000/testing/CROZ/'
opt.step = 64
opt.size = 386
opt.patch_fold_A = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+ '/A/'
opt.patch_fold_B = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+'/B/'
A_fold = opt.im_fold + 'A/'
B_fold = opt.im_fold +  'B/'

opt.input_nc =3
sdmkdir(opt.patch_fold_A)
sdmkdir(opt.patch_fold_B)
imlist=[]
imnamelist=[]

for root,_,fnames in sorted(os.walk(A_fold)):
    for fname in fnames:
        if fname.endswith('.png') and "M1BS" in fname:
            path = os.path.join(root,fname)
            path_mask = os.path.join(B_fold,fname)
            imlist.append((path,path_mask,fname))
            imnamelist.append(fname)
for im_path,mask_path,imname in  imlist:
    png = misc.imread(im_path,mode='RGB')
    print mask_path
    mask = misc.imread(mask_path)
    print mask.shape
    w,h,z = png.shape
    savepatch_train(png,mask,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname,opt.patch_fold_B+'/'+imname)
