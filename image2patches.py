
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
opt = TestOptions().parse()
#opt = argparse.ArgumentParser().parse_args()
#opt.im_fold='/gpfs/projects/LynchGroup/Train_all/CROPPED/p1000/'
opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p300/'
opt.im_fold ='/nfs/bigbox/hieule/penguin_data/Test/CROZ/CROPPED/p300/'
opt.step = 128
opt.size = 256
opt.patch_fold_A = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+ '/A/'
opt.patch_fold_B = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+'/B/'
opt.name = 'train_on_p300_2'
opt.which_epoch = 10
opt.patch_fold_res = opt.im_fold + 'PATCHES/res/' + opt.name+ '/'
opt.im_res = opt.im_fold + 'res/' + opt.name +'e'+str(opt.which_epoch)+'/'
#opt.checkpoints_dir = '/nfs/bigbox/hieule/p1000/'+ 'trainPATCHES/'+ 'checkpoints/'
opt.checkpoints_dir = '/nfs/bigbox/hieule/penguin_data/checkpoints/'

A_fold = opt.im_fold + 'A/'
B_fold = opt.im_fold +  'B/'

sdmkdir(opt.patch_fold_A)
sdmkdir(opt.patch_fold_B)
sdmkdir(opt.patch_fold_res)
sdmkdir(opt.im_res)
imlist=[]
imnamelist=[]
opt.no_dropout = True
model = create_model(opt)
print(model)

for root,_,fnames in sorted(os.walk(A_fold)):
    for fname in fnames:
        if fname.endswith('.png') and "M1BS" in fname:
            path = os.path.join(root,fname)
            path_mask = os.path.join(B_fold,fname)
            imlist.append((path,path_mask,fname))
            imnamelist.append(fname)
for im_path,mask_path,imname in  imlist:
    png = misc.imread(im_path,mode='RGB')
    w,h,z = png.shape
    #savepatch_test(png,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname)
    #savepatch_train(png,mask,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname,opt.patch_fold_B+'/'+imname)

for root,_,fnames in sorted(os.walk(opt.patch_fold_A)):
    for fname in fnames:
        if fname.endswith('png'):
            print fname
            im = misc.imread(os.path.join(root,fname))
            im = np.transpose(im,(2,0,1))
            imtensor = torch.from_numpy(im).float().div(255)
            imtensor = imtensor - 0.5
            imtensor = imtensor * 2
            imtensor = torch.unsqueeze(imtensor,0)
            temp = model.get_prediction(imtensor)['raw_out'][:,:,0]
            misc.toimage(temp,mode='L').save(os.path.join(opt.patch_fold_res,fname))

for im_path,mask_path,imname in  imlist:
    png = misc.imread(im_path,mode='RGB')
    w,h,z = png.shape
    mask = patches2png(opt.patch_fold_res,imname,w,h,opt.step,opt.size)
    misc.toimage(mask.astype(np.uint8),mode='L').save(os.path.join(opt.im_res,imname))
visABC(opt.im_fold,opt.name+'e'+str(opt.which_epoch))