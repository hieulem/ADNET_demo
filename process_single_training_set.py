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

padding = 1000
training_fold =  '/gpfs/projects/LynchGroup/CROZtrain/CROPPED/p'+str(padding)+'/training/'
root_fold = '/gpfs/projects/LynchGroup/CROZtest/'
if not os.path.isdir(training_fold):
    os.makedirs(training_fold)

gt_filename = root_fold + 'AnnotatedGuano/crozier_guanoarea_stereographic'
basename = 'WV02_20110131195115_1030010009CCF900_11JAN31195115-M1BS-052549143040_01_P003_u08rf3031'
im_filename = root_fold + basename 
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
print(crop_image.shape)

index = np.nonzero(mask)
maxx = np.max(index[0])
minx = np.min(index[0])
maxy = np.max(index[1])
miny = np.min(index[1])
print(maxx,minx,maxy,miny)

save_mask = mask[minx-padding:maxx+padding,miny-padding:maxy+padding]

Image.fromarray(save_mask.astype(np.uint8)).save(training_fold+'/'+basename+'.png',cmap=cm.gray)

plt.figure(1)
plt.imshow(save_mask)

tifimg = dataset.read()
savetif = tifimg[:,minx-padding:maxx+padding,miny-padding:maxy+padding]
outtif = training_fold + basename+'.npy'
plt.figure(2)
plt.imshow(savetif[1,:,:],cmap=cm.gray)
np.save(outtif,savetif)

"""
out_meta.update({"driver": "GTiff",
                 "height": crop_image.shape[1],
                 "width": crop_image.shape[2],
                 "transform": mt})

with rasterio.open(outtif, "w", **out_meta) as dest:
    dest.write(crop_image)

cropped = rasterio.open(outtif)
crop_masked_image,ctr = rasterio.mask.mask(cropped,[feature["geometry"] for feature in transformed_gt])
print ctr

band = []
band2 = []
band3 = []
for i in range(2,5):
    band.append(np.squeeze(masked_image[i,:,:]))
    band2.append(np.squeeze(crop_image[i,:,:]))
    band3.append(np.squeeze(crop_masked_image[i,:,:]))
plt.figure(3)
plt.imshow(np.dstack((band[0],band[1],band[2])))


plt.figure(2)
plt.imshow(np.dstack((band2[0],band2[1],band2[2])))

plt.figure(100)
plt.imshow(np.dstack((band3[0],band3[1],band3[2])))
"""
plt.show()



