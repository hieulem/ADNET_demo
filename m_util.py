import os
import numpy as np
from PIL import Image
from scipy import misc

def convertMbandstoRGB(tif,imname):
    if tif.shape[0] ==1:
        return tif
    if "QB" in imname:
        return tif[(3,2,1),:,:]
    if "WV" in imname:
        if tif.shape[0] ==8:
            return tif[(5,3,2),:,:]
        if tif.shape[0] ==4:
            return tif[(3,2,1),:,:]
    if "IK" in imname:
        return tif[(3,2,1),:,:]

def to_rgb3b(im):
    # as 3a, but we add an extra copy to contiguous 'C' order
    # data
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')
def sdsaveim(savetif,name):
    print savetif.shape
    if savetif.dtype == np.uint16:
        savetif = savetif.astype(np.float)
        for i in range(0,savetif.shape[2]):
            savetif[:,:,i] =  savetif[:,:,i] / np.max(savetif[:,:,i]) * 255
        savetif = savetif.astype(np.uint8) 
        print np.max(np.max(savetif,axis=0),axis=0)
    if savetif.shape[2] == 3:
        Image.fromarray(savetif.astype(np.uint8)).save(name)
    if savetif.shape[2] == 1:
        Image.fromarray(np.squeeze(savetif.astype(np.uint8)),mode='L').save(name)
    #plt.imshow(savetif[1,:],cmap=cm.gray)

def sdmkdir(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def png2patches(png,step,size):
    step = np.int32(step)
    size=  np.int32(size)
    z,w,h = png.shape
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)

    patches = np.zeros((ni,nj,z,size,size))
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            patches[i,j,:,:,:] = png[i*step:i*step+size,j*step:j*step+size,:]
    for i in range(0,ni-1):
        patches[i,nj-1,:,:,:] = png[i*step:i*step+size,h-size:h,:]

    for j in range(0,nj-1):
        patches[ni-1,j,:,:,:] = png[w-size:w,j*step:j*step+size,:]
    patches[ni-1,nj-1,:,:,:] = png[w-size:w,h-size:h,:]
    return patches


def patches2png(patches,w,h,step,size):
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
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(basename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(basename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')


def savepatch_test_with_mask(png,mask,w,h,step,size,imbasename,patchbasename):

    ni = np.int32(np.floor((w- size)/step) +2)
    nj = np.int32(np.floor((h- size)/step) +2)

    for i in range(0,ni-1):
        for j in range(0,nj-1):
            name = format(i,'03d')+'_'+format(j,'03d')+'.png'
            m = mask[i*step:i*step+size,j*step:j*step+size]
            misc.toimage(m,mode='L').save(patchbasename+name)
            misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(imbasename+name)
    for i in range(0,ni-1):
        
        name = format(i,'03d')+'_'+format(nj-1,'03d')+'.png'
        m = mask[i*step:i*step+size,h-size:h]
        misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(imbasename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


    for j in range(0,nj-1):
        name = format(ni-1,'03d')+'_'+format(j,'03d')+'.png'
        m = mask[w-size:w,j*step:j*step+size]
        misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(imbasename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
    
    m= mask[w-size:w,h-size:h]
    misc.toimage(m,mode='L').save(patchbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(imbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')

def savepatch_train(png,mask,w,h,step,size,imbasename,patchbasename):

    ni = np.int32(np.floor((w- size)/step) +2)
    nj = np.int32(np.floor((h- size)/step) +2)

    for i in range(0,ni-1):
        for j in range(0,nj-1):
            name = format(i,'03d')+'_'+format(j,'03d')+'.png'
            m = mask[i*step:i*step+size,j*step:j*step+size]
            if m.any():
                misc.toimage(m,mode='L').save(patchbasename+name)
            misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(imbasename+name)
    for i in range(0,ni-1):
        
        name = format(i,'03d')+'_'+format(nj-1,'03d')+'.png'
        m = mask[i*step:i*step+size,h-size:h]
        if m.any():
            misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(imbasename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


    for j in range(0,nj-1):
        name = format(ni-1,'03d')+'_'+format(j,'03d')+'.png'
        m = mask[w-size:w,j*step:j*step+size]
        if m.any():
            misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(imbasename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
    
    m= mask[w-size:w,h-size:h]
    if m.any():
        misc.toimage(m,mode='L').save(patchbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(imbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')



def patches2png(patch_fold,imname,w,h,step,size):
    png = np.zeros((w,h))
    ws = np.zeros((w,h))
    
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)
    print ni,nj    
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            
            patch = misc.imread(patch_fold + '/' + imname + format(i,'03d')+'_'+format(j,'03d')+'.png',mode='L')
            patch[patch<50] = 0
            png[i*step:i*step+size,j*step:j*step+size]=  png[i*step:i*step+size,j*step:j*step+size]+patch 
            ws[i*step:i*step+size,j*step:j*step+size]=  ws[i*step:i*step+size,j*step:j*step+size]+ 1
    for i in range(0,ni-1):
        patch = misc.imread(patch_fold + '/' + imname + format(i,'03d')+'_'+format(nj-1,'03d')+'.png',mode='L')
        patch[patch<50] = 0
        png[i*step:i*step+size,h-size:h] =  png[i*step:i*step+size,h-size:h]+ patch
        ws[i*step:i*step+size,h-size:h] =  ws[i*step:i*step+size,h-size:h]+ 1

    for j in range(0,nj-1):
        patch = misc.imread(patch_fold + '/' + imname + format(ni-1,'03d')+'_'+format(j,'03d')+'.png',mode='L')
        patch[patch<50] = 0
        png[w-size:w,j*step:j*step+size]= png[w-size:w,j*step:j*step+size]+ patch
        ws [w-size:w,j*step:j*step+size]= ws [w-size:w,j*step:j*step+size]+ 1
   
    patch = misc.imread(patch_fold + '/' + imname + format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png',mode='L')
    patch[patch<50] = 0
    png[w-size:w,h-size:h] = png[w-size:w,h-size:h]+ patch
    ws [w-size:w,h-size:h] = ws [w-size:w,h-size:h]+ 1
    png = np.divide(png,ws)
    
    return png

