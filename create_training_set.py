import rasterio
from rasterio import mask,features,warp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path
import fiona
import numpy as np
from osgeo import gdal,osr
from shapely.geometry import shape,mapping
from shapely.geometry.polygon import LinearRing,Polygon
from mfuncshape import BoundingBoxShape,ReadProjection,TransformShape
from PIL import Image
from matplotlib import cm
from baseoption import BaseOptions
from util import sdmkdir, convertMbandstoRGB,sdsaveim

opt = BaseOptions().parse()
opt.padding = 1000
opt.root = '/gpfs/projects/LynchGroup/'
opt.raw_fold = opt.root + 'Train_all/raw/'
opt.tif_fold = opt.root + 'Orthoed/'
opt.training_fold = opt.root + 'Train_all/CROPPED/p1000/'
opt.A = opt.training_fold + 'A/'
opt.B = opt.training_fold + 'B/'

opt.visdir = opt.root + 'Train_all/CROPPED/p1000/vis/'
sdmkdir(opt.training_fold)
sdmkdir(opt.A)
sdmkdir(opt.B)
sdmkdir(opt.visdir)


def full_frame(width=None, height=None):
    matplotlib.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig
alldata = [i for i in next(os.walk(opt.raw_fold))[1] if not i.startswith('.')]
print alldata

for idx in range(0,len(alldata)):
    root_fold = opt.raw_fold+alldata[idx] +'/'
    gta = [i for i in next(os.walk(root_fold))[2] if i.endswith('.shp')]
    txt = [i for i in next(os.walk(root_fold))[2] if i.endswith('.txt')]
    a = open(os.path.join(root_fold,txt[0]),'r')
    name = a.read()
    gt_prj = ReadProjection(root_fold+gta[0].replace('.shp','.prj'))
    gt = fiona.open(root_fold+gta[0])
    
    for BAND in ['P','M']:
        imname= name.replace('{X}',BAND)
        TIFim = opt.tif_fold+imname+'.tif'

        dataset = rasterio.open(TIFim)
        im_prj = ReadProjection(TIFim.replace('.tif','.prj'))


        coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
        transformed_gt,bb = TransformShape(gt,coordTrans,padding = opt.padding)

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

        save_mask = mask[minx-opt.padding:maxx+opt.padding,miny-opt.padding:maxy+opt.padding]

        Image.fromarray(save_mask.astype(np.uint8)).save(opt.B+'/'+imname+'.png',cmap=cm.gray)

        plt.figure(1)
        plt.imshow(save_mask)

        tifimg = dataset.read()
        
        tifimg = convertMbandstoRGB(tifimg,imname)
        print tifimg.shape
        savetif = tifimg[:,minx-opt.padding:maxx+opt.padding,miny-opt.padding:maxy+opt.padding]
        outtif = opt.training_fold + imname+'.npy'
                
        savetif = np.transpose(savetif,(1,2,0))
        sdsaveim(savetif,opt.A+'/'+imname+'.png')
