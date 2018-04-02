from models.models import create_model
from data.png_dataset import PngDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
from data.data_loader import CreateDataLoader
import torch
import os.path
import argparse
from scipy import misc
from m_util import *
from vis import *
import cv2
opt = TestOptions().parse()

#opt = argparse.ArgumentParser().parse_args()
#opt.im_fold='/gpfs/projects/LynchGroup/Train_all/CROPPED/p1000/'
#opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p300/'
#opt.im_fold ='/nfs/bigbox/hieule/penguin_data/Test/CROZ/CROPPED/p300/'
opt.im_fold_temp ='/nfs/bigbox/hieule/penguin_data/Test/*TEST*/CROPPED/p300/'
#opt.im_fold ='/nfs/bigbox/hieule/penguin_data/Test/CROZ/CROPPED/p300/'
for t in ["PAUL","CROZ"]:
    opt.im_fold = opt.im_fold_temp.replace("*TEST*",t)
    opt.input_nc =4
    opt.model = 'single_unet_4c'
    opt.step = 64
    opt.size = 256
    opt.patch_fold_A = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+ '/A/'
    opt.patch_fold_B = opt.im_fold+'PATCHES/'+str(opt.step)+'_'+ str(opt.size)+'/B/'
    opt.name = '4c_train_on_p300_3'
    opt.which_epoch = 200
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
        mask = misc.imread(mask_path,mode='L')
        savepatch_test_with_mask(png,mask,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname,opt.patch_fold_B+'/'+imname)
        #savepatch_test(png,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname)
        #savepatch_train(png,mask,w,h,opt.step,opt.size,opt.patch_fold_A+'/'+imname,opt.patch_fold_B+'/'+imname)

    for root,_,fnames in sorted(os.walk(opt.patch_fold_A)):
        for fname in fnames:
            if fname.endswith('png'):
                print fname
                im = misc.imread(os.path.join(root,fname))
                im = np.transpose(im,(2,0,1))
                imtensor = torch.from_numpy(im).float().div(255)
                B_img = misc.imread(os.path.join(opt.patch_fold_B,fname),mode='L')
                
                C_img = np.copy(B_img).astype(np.uint8)
                C_img = cv2.dilate(C_img, np.ones((30,30)))
                C_img[C_img>0] = 255
                C_img = np.expand_dims(C_img, axis=0)
                C_img = torch.from_numpy(C_img).float().div(255)
                imtensor = imtensor - 0.5
                imtensor = imtensor * 2
                imtensor = torch.unsqueeze(imtensor,0)
                C_img = torch.unsqueeze(C_img,0)
                input= {'A':imtensor,'C':C_img}
                temp = model.get_prediction(input)['raw_out'][:,:,0]
                misc.toimage(temp,mode='L').save(os.path.join(opt.patch_fold_res,fname))

    for im_path,mask_path,imname in  imlist:
        png = misc.imread(im_path,mode='RGB')
        w,h,z = png.shape
        mask = patches2png(opt.patch_fold_res,imname,w,h,opt.step,opt.size)
        misc.toimage(mask.astype(np.uint8),mode='L').save(os.path.join(opt.im_res,imname))
    visABC(opt.im_fold,opt.name+'e'+str(opt.which_epoch))
    visTIF(opt.im_fold,opt.name+'e'+str(opt.which_epoch))
