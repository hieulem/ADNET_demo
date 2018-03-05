import os.path

import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
class TifDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.GTroot = opt.dataroot
        self.imlist = []
        self.imname = []
        for root,_,fnames in sorted(os.walk(self.root)):
            for fname in fnames:
                if fname.endswith('.npy'):
                    path = os.path.join(root,fname)
                    self.imlist.append(path)
                    self.imname.append(fname)
        self.tifimg = np.load(self.imlist[0]).astype(np.uint8)
         
        self.tifimg = np.mean(self.tifimg,axis=0,keepdims=True)
        
        GTmask = np.asarray(Image.open(os.path.join(self.GTroot,self.imname[0][:-4]+'.png')))
        self.GTmask = np.zeros((1,GTmask.shape[0],GTmask.shape[1]),dtype=np.uint8)
        self.GTmask[0] = GTmask.astype(np.uint8)
        z,w,h = self.tifimg.shape
        self.nx=np.floor(w/256).astype(np.uint8)
        self.ny=np.floor(h/256).astype(np.uint8)
    
    def __len__(self):
        return 10000
    def name(self):
        return 'TIFDATASET'
    
    def getpatch(self,i,j):
        A_img = self.tifimg[:,i*256:(i+1)*256,j*256:(j+1)*256]
        B_img = self.GTmask[:,i*256:(i+1)*256,j*256:(j+1)*256]
        A_img = torch.from_numpy(A_img).float().div(255)
        B_img = torch.from_numpy(B_img).float().div(255)
        
        A_img = torch.unsqueeze(A_img,0)
        B_img = torch.unsqueeze(B_img,0)
        return  {'A': A_img, 'B': B_img,'imname':self.imname[0]}
    def get_number_of_patches(self):
        return self.nx,self.ny
    def __getitem__(self,index):
        z,w,h = self.tifimg.shape
        w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
        h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
        A_img = self.tifimg[:, w_offset:w_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize] 
        B_img = self.GTmask[:,w_offset:w_offset + self.opt.fineSize, h_offset:h_offset + self.opt.fineSize]
        A_img = torch.from_numpy(A_img).float().div(255)
        B_img = torch.from_numpy(B_img).float().div(255)
        
        return  {'A': A_img, 'B': B_img}
