import rasterio
from rasterio import mask,features,warp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path
import fiona
import numpy as np
import osgeo
from osgeo import gdal,osr
from shapely.geometry import shape,mapping
from shapely.geometry.polygon import LinearRing,Polygon
from mfuncshape import BoundingBoxShape,ReadProjection,TransformShape
from PIL import Image
from matplotlib import cm
from baseoption import BaseOptions
from m_util import sdmkdir, convertMbandstoRGB,sdsaveim
import pandas as pd
from shutil import copyfile
opt = BaseOptions().parse()
opt.padding = 300
#file= '/gpfs/projects/LynchGroup/CatalogIDs_training_shapefiles.xlsx'
file = '/gpfs/projects/LynchGroup/CROZ_IDs_Test.xlsx'
opt.root = '/gpfs/projects/LynchGroup/'
opt.fold = 'Test/CROZ/'

opt.raw_fold = opt.root + opt.fold+ '/raw/'
opt.tif_fold = opt.root + 'Orthoed/'
opt.training_fold = opt.root + opt.fold+ '/CROPPED/p'+str(opt.padding)+'/'
opt.A = opt.training_fold + 'A/'
opt.B = opt.training_fold + 'B/'

opt.visdir = opt.root + opt.fold+ '/CROPPED/p' +str(opt.padding)+ '/vis/'
opt.ctifdir = opt.root + opt.fold+ '/CROPPED/p' +str(opt.padding)+ '/tif/'
sdmkdir(opt.training_fold)
sdmkdir(opt.A)
sdmkdir(opt.B)
sdmkdir(opt.visdir)
sdmkdir(opt.ctifdir)

#shape_dir= '/gpfs/projects/LynchGroup/Colony\ shapefiles\ from\ imagery/'
shape_dir = opt.root+ '/Annotated_shapefiles/'

anno = pd.read_excel(file,sheet_name=0)
tif = anno['Filename']
shape =  anno['Shapefile of guano']
for i in range(0,len(tif)):
    name= tif[i].encode('ascii','ignore')
    gta= shape[i].encode('ascii','ignore')
    name = name.replace('.tif','') 
    gt_prj = ReadProjection(shape_dir+gta+'.prj')
    gt = fiona.open(shape_dir+gta+'.shp')
    
    name= name.replace('-M','-*****')
    name= name.replace('-P','-*****')
    for BAND in ['-P','-M']:
        imname= name.replace('-*****',BAND)
        print imname
        TIFim = opt.tif_fold+imname+'.tif'
        if os.path.isfile(TIFim):
            dataset = rasterio.open(TIFim)
            
            #im_prj = ReadProjection(TIFim.replace('.tif','.prj'))
            #print dataset.crs
            #print dataset.crs.to_string()
            #print dataset.crs.wkt
            #print im_prj
            im_prj = osgeo.osr.SpatialReference()
            im_prj.ImportFromWkt(dataset.crs.wkt)
            #print im_prj2
            coordTrans = osr.CoordinateTransformation(gt_prj,im_prj)
            transformed_gt,bb = TransformShape(gt,coordTrans,padding = opt.padding)

            crop_image, ct = rasterio.mask.mask(dataset,[po['geometry'] for po in bb],crop=True)

            out_meta = dataset.meta.copy()
            out_meta.update({"driver":"GTiff",
                            "height": crop_image.shape[1],
                            "width": crop_image.shape[2],
                            "transform": ct
                            })
            with rasterio.open(opt.ctifdir+imname+'.tif',"w",**out_meta) as dest:
                dest.write(crop_image)
            dest = rasterio.open( opt.ctifdir+imname+'.tif')
            
            masked_image, mt = rasterio.mask.mask(dest,[feature["geometry"] for feature in transformed_gt])
            mask = masked_image.mean(axis=0)
            mask[mask>0]=255
            mask[mask<255]= 0 
            Image.fromarray(mask.astype(np.uint8)).save(opt.B+'/'+imname+'.png',cmap=cm.gray)
            tifimg = crop_image
            tifimg = convertMbandstoRGB(tifimg,imname)
            savetif = tifimg
            savetif = np.transpose(savetif,(1,2,0))
            sdsaveim(savetif,opt.A+'/'+imname+'.png')
            #copyfile(opt.tif_fold+imname+'.prj',opt.A+'/'+imname+'.prj')
            #copyfile(opt.tif_fold+imname+'.tif',ndir+imname+'.tif')
            #copyfile(opt.tif_fold+imname+'.prj',ndir+imname+'.prj')

