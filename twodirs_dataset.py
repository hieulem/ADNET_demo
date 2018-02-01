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
from scipy import signal
import time
class TwodirsDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        print self.dir_A
        self.A_paths,self.imname = make_dataset(self.dir_A)
        #self.B_paths = make_dataset(self.dir_B)

       # self.A_paths = sorted(self.A_paths)
        #self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transformA = transforms.Compose(transform_list)
       # self.transformA = transforms.Compose([transforms.ToTensor()])  
        self.transformB = transforms.Compose([transforms.ToTensor()])
    def __getitem2__(self, index):
        return getitemxy(self,index,-1,-1)
    def __getitem__(self,index):

        A_path = self.A_paths[index % self.A_size]
        imname = self.imname[index % self.A_size]
        
        index_A = index % self.A_size
        #index_B = random.randint(0, self.B_size - 1)
        #B_path = self.B_paths[index_B]
        B_path = os.path.join(self.dir_B,imname.replace('.jpg','.png'))
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B,imname.replace('.jpg','.jpg'))

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        w = np.float(A_img.size[0])
        h = np.float(A_img.size[1])
        if os.path.isfile(B_path): 
            B_img = Image.open(B_path)
        else:
            B_img = Image.fromarray(np.zeros((int(w),int(h)),dtype = np.uint8),mode='L')
        #print B_img.size()
        if self.opt.useshadowbd:
            grad1 = signal.convolve2d(np.asarray(B_img).astype(np.float),[[1,1],[-1,-1]],boundary='symm',mode ='same')
            grad2 = signal.convolve2d(np.asarray(B_img),[[-1,1],[-1,1]],boundary='symm',mode ='same')
            grad1[grad1!=0]= 255
            grad2[grad2!=0]=255
            grad3 = grad1+grad2
            shadow_boundary = signal.convolve2d(grad3,[[1,1,1],[1,1,1],[1,1,1]],boundary='symm',mode= 'same')
            shadow_boundary[shadow_boundary!=0] = 255
            shadow_boundary = Image.fromarray(shadow_boundary.astype(np.uint8))
   #     time.sleep(5)
    


      #print A_path
        #print imname
        #print B_path
        if self.opt.randomSize:
            self.opt.loadSize = np.random.randint(257,300,1)[0]
        if self.opt.keep_ratio:
            if w>h:
                ratio = np.float(self.opt.loadSize)/np.float(h)
                neww = np.int(w*ratio)
                newh = self.opt.loadSize
            else:
                ratio = np.float(self.opt.loadSize)/np.float(w)
                neww = self.opt.loadSize
                newh = np.int(h*ratio)
        else:
            neww = self.opt.loadSize
            newh = self.opt.loadSize
        
        if self.opt.tsize:
            neww = self.opt.tw
            newh = self.opt.th
        
        A_img = A_img.resize((neww, newh),Image.NEAREST)
        B_img = B_img.resize((neww, newh),Image.NEAREST)
        if self.opt.useshadowbd:
            shadow_boundary = shadow_boundary.resize((neww,newh),Image.NEAREST)
        
        w = A_img.size[0]
        h = A_img.size[1]
        if not self.opt.not_use_log and not self.opt.use_log_01:
            A_img = np.asarray(A_img).astype(np.double) + 1
            A_img = np.log(A_img)
            A_img = (A_img-np.log(1))/(np.log(256) - np.log(1)) *255

        A_img = self.transformA(A_img)
        B_img = self.transformB(B_img)
        #B_sm_img = self.transformB(B_sm_img)

        #print shadow_boundary
        #shadow_boundary.show()
        if self.opt.useshadowbd:
            shadow_boundary = self.transformB(shadow_boundary)

        if not self.opt.no_crop:        
            w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
            h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
                
            A_img = A_img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B_img = B_img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]      
              
            if self.opt.useshadowbd:
                shadow_boundary = shadow_boundary[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]      
        
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
            if self.opt.useshadowbd:
                shadow_boundary = shadow_boundary.index_select(2,idx)
        if self.opt.useshadowbd:
            return {'A': A_img, 'B': B_img,'sdbr': shadow_boundary,
                'A_paths': A_path, 'B_paths': B_path,'imname':imname}
        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path,'imname':imname}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TwodirsDataset'
