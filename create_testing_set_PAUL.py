import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path
from resizeimage import resizeimage
import rasterio.features
import rasterio.warp
import fiona
import numpy as np
from osgeo import gdal,osr
from shapely.geometry import shape,mapping
from shapely.geometry.polygon import LinearRing,Polygon
from mfuncshape import BoundingBoxShape,ReadProjection,TransformShape
from PIL import Image
from matplotlib import cm

from util import sdmkdir,convertMbandstoRGB
padding=1000
btraining_fold =  '/gpfs/projects/LynchGroup/Train_all/PAULtrain/CROPPED/p'+str(padding)+'/'
testing_fold_im =  '/gpfs/projects/LynchGroup/Train_all/PAULtrain/CROPPED/p'+str(padding)+'/A/'
testing_fold_mask =  '/gpfs/projects/LynchGroup/Train_all/PAULtrain/CROPPED/p'+str(padding)+'/B/'
sdmkdir(testing_fold_im)
sdmkdir(testing_fold_mask)
root_fold = '/gpfs/projects/LynchGroup/PAULtest/'
gt_filename = root_fold + 'AnnotatedGuano/paulet_guanoarea_stereographic'
listallname = []
for root,_,files in os.walk(root_fold):
    for f in files:
        if f.endswith('.prj'):
            #with rasterio.open(root_fold+f) as t:
            #    print t.count
            #    if t.count ==4:
            listallname.append(f[:-4])
print(listallname)
print len(listallname)
for i in range(0,len(listallname)):
    imname =listallname[i]
    print imname
    im_filename = root_fold + imname 
    gt_prj = ReadProjection(gt_filename+'.prj')
    im_prj = ReadProjection(im_filename+'.prj')

    TIFim = im_filename+ '.tif'
    dataset = rasterio.open(TIFim)

    coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
    gt = fiona.open(gt_filename+'.shp')
    transformed_gt,bb = TransformShape(gt,coordTrans,padding = padding)

    masked_image, mt = rasterio.mask.mask(dataset,[feature["geometry"] for feature in transformed_gt])
    crop_image, ct = rasterio.mask.mask(dataset,[po['geometry'] for po in bb])

    out_meta = dataset.meta.copy()



    mask = masked_image.mean(axis=0)
    mask[mask>0]=255
    mask[mask<255]= 0 

    index = np.nonzero(mask)
    maxx = np.max(index[0])
    minx = np.min(index[0])
    maxy = np.max(index[1])
    miny = np.min(index[1])

    save_mask = mask[minx-padding:maxx+padding,miny-padding:maxy+padding]

    Image.fromarray(save_mask.astype(np.uint8)).save(testing_fold_mask+'/'+imname+'.png',cmap=cm.gray)

    #plt.figure(1)
    #plt.imshow(save_mask)

    tifimg = dataset.read()
    print tifimg.shape
    tifimg = convertMbandstoRGB(tifimg,imname)
    
    savetif = tifimg[:,minx-padding:maxx+padding,miny-padding:maxy+padding]
    savetif = np.transpose(savetif,(1,2,0))
    print savetif.dtype
    if savetif.dtype == np.uint16:
        savetif = (savetif.astype(np.float) / np.max(savetif)*255).astype(np.uint8) 
    print np.max(savetif)
    if savetif.shape[2] == 3:
        Image.fromarray(savetif.astype(np.uint8)).save(testing_fold_im+'/'+imname+'.png')
    if savetif.shape[2] == 1:
        Image.fromarray(np.squeeze(savetif.astype(np.uint8)),mode='L').save(testing_fold_im+'/'+imname+'.png')
    
'''
    Image.fromarray(savetif[:,:,(3,2,1)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'321.png')
    Image.fromarray(savetif[:,:,(3,2,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'320.png')
    Image.fromarray(savetif[:,:,(3,1,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'312.png')
    Image.fromarray(savetif[:,:,(3,1,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'310.png')
    Image.fromarray(savetif[:,:,(3,0,1)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'301.png')
    Image.fromarray(savetif[:,:,(3,0,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'302.png')
    Image.fromarray(savetif[:,:,(2,1,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'210.png')
    Image.fromarray(savetif[:,:,(2,1,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'213.png')
    Image.fromarray(savetif[:,:,(2,3,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'230.png')
    Image.fromarray(savetif[:,:,(2,3,1)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'231.png')
    Image.fromarray(savetif[:,:,(2,0,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'203.png')
    Image.fromarray(savetif[:,:,(2,0,1)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'201.png')
    Image.fromarray(savetif[:,:,(1,2,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'123.png')
    Image.fromarray(savetif[:,:,(1,2,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'120.png')
    Image.fromarray(savetif[:,:,(1,3,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'132.png')
    Image.fromarray(savetif[:,:,(1,3,0)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'130.png')
    Image.fromarray(savetif[:,:,(1,0,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'102.png')
    Image.fromarray(savetif[:,:,(1,0,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'103.png')
    Image.fromarray(savetif[:,:,(0,1,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'012.png')
    Image.fromarray(savetif[:,:,(0,1,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'013.png')
    Image.fromarray(savetif[:,:,(0,2,3)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'023.png')
    Image.fromarray(savetif[:,:,(0,2,1)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'021.png')
    Image.fromarray(savetif[:,:,(0,3,2)].astype(np.uint8)).save(testing_fold_im+'/'+imname+'032.png')
'''
