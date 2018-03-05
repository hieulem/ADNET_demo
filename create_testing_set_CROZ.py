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


btraining_fold =  '/gpfs/projects/LynchGroup/CROZtrain/CROPPED/p'+str(padding)+'/'
testing_fold_im =  '/gpfs/projects/LynchGroup/CROZtrain/CROPPED/p'+str(padding)+'/A/'
testing_fold_mask =  '/gpfs/projects/LynchGroup/CROZtrain/CROPPED/p'+str(padding)+'/B/'

sdmkdir(testing_fold_im)
sdmkdir(testing_fold_mask)

root_fold = '/gpfs/projects/LynchGroup/CROZtest/'
gt_filename = root_fold + 'AnnotatedGuano/crozier_guanoarea_stereographic'
listallname = []
for root,_,files in os.walk(root_fold):
    for f in files:
        if f.endswith('.prj'):
                listallname.append(f[:-4])
print(listallname)
for i in range(0,len(listallname)):
    imname =listallname[i]
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
    print(mask.shape)

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
    #plt.imshow(savetif[1,:],cmap=cm.gray)
