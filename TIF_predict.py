from models.models import create_model
from data.tif_dataset import TifDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
import time
from data.data_loader import CreateDataLoader
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import torch
import os.path

def makedirifnotexist(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def full_frame(width=None, height=None):
    matplotlib.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig

opt = TestOptions().parse()
opt.root = '/gpfs/projects/LynchGroup/CROZtrain/'
opt.dataset_mode='tif'
opt.fineSize = 256
opt.model ='single_unet'
opt.display_port = 9998
opt.resdir = opt.root+'/results/'+opt.name+'/'
opt.visdir = opt.root+'/vis/'+opt.name+'/'
opt.checkpoints_dir = opt.root + '/checkpoints'
opt.gpu_ids= [0]
opt.step = 64
opt.size = 256
   
print(opt.resdir)
print(opt.root)
makedirifnotexist(opt.resdir)
makedirifnotexist(opt.visdir)

def totensor(A_img):
    A_img = torch.from_numpy(A_img).float().div(255)
    A_img = torch.unsqueeze(A_img,0)
    return A_img

def tif2patches(tif,step,size):
    step = np.int32(step)
    size=  np.int32(size)
    z,w,h = tif.shape
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)

    patches = np.zeros((ni,nj,z,size,size))
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            patches[i,j,:,:,:] = tif[:,i*step:i*step+size,j*step:j*step+size]
            #print i*step,i*step+size
    for i in range(0,ni-1):
        patches[i,nj-1,:,:,:] = tif[:,i*step:i*step+size,h-size:h]

    for j in range(0,nj-1):
        patches[ni-1,j,:,:,:] = tif[:,w-size:w,j*step:j*step+size]
    patches[ni-1,nj-1,:,:,:] = tif[:,w-size:w,h-size:h]
    return patches

def patches2tif(patches,w,h,step,size):
    tif = np.zeros((1,w,h))
    ws = np.zeros((1,w,h))
    
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)
    
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            tif[:,i*step:i*step+size,j*step:j*step+size]=  tif[:,i*step:i*step+size,j*step:j*step+size]+ patches[i,j,:,:,:]
            ws[:,i*step:i*step+size,j*step:j*step+size]=  ws[:,i*step:i*step+size,j*step:j*step+size]+ 1
           
    for i in range(0,ni-1):
        tif[:,i*step:i*step+size,h-size:h] =  tif[:,i*step:i*step+size,h-size:h]+ patches[i,nj-1,:,:,:] 
        ws[:,i*step:i*step+size,h-size:h] =  ws[:,i*step:i*step+size,h-size:h]+ 1

    for j in range(0,nj-1):
        tif[:,w-size:w,j*step:j*step+size]= tif[:,w-size:w,j*step:j*step+size]+ patches[ni-1,j,:,:,:]
        ws[:,w-size:w,j*step:j*step+size]= ws[:,w-size:w,j*step:j*step+size]+ 1
   
    tif[:,w-size:w,h-size:h] = tif[:,w-size:w,h-size:h]+ patches[ni-1,nj-1]
    ws[:,w-size:w,h-size:h] = ws[:,w-size:w,h-size:h]+ 1
    
    tif = np.divide(tif,ws)


    return tif
        

model = create_model(opt)
for k in range(0,13):

    opt.dataroot = opt.root+'/CROPPED/p1000/'+str(k)+'/'
    imlist=[]
    imnamelist=[]
    for root,_,fnames in sorted(os.walk(opt.dataroot)):
        for fname in fnames:
            if fname.endswith('.npy'):
                path = os.path.join(root,fname)
                imlist.append(path)
                imnamelist.append(fname)
    print(imlist)

    tif = np.load(imlist[0]).astype(np.uint8)
    z,w,h = tif.shape
    dd = tif2patches(np.copy(tif),opt.step,opt.size)
    s = np.asarray(dd.shape)
    print(s)
    s[2]=1
    out = np.zeros(s)
    print(out.shape)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            t = dd[i,j,:,:,:]
            input = totensor(t)
            temp = model.get_prediction(input)
            out[i,j,:,:,:] = temp['raw_out'][:,:,0]

    tt =  patches2tif(out,w,h,opt.step,opt.size)
    #fi1 = plt.figure(1)
    fi1 = full_frame()
    colormap = plt.cm.viridis
    plt.imshow(np.squeeze(tt),colormap)
    plt.axis('off')
    fi1.savefig(opt.visdir+imnamelist[0]+'_pred.png',bbox_inches='tight',pad_inches=0,dpi=1000)
    #fi2 = plt.figure(2)
    fi2 = full_frame()
    plt.imshow(np.squeeze(tif[4,:,:]),colormap)
    plt.axis('off')
    fi2.savefig(opt.visdir+imnamelist[0]+'_ori.png',bbox_inches='tight',pad_inches=0,dpi=1000)
    plt.show()
