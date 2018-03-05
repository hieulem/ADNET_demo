from shapely.geometry import shape
import fiona 
from PIL import Image 
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from resizeimage import resizeimage
import glob
folder =  "/gpfs/projects/LynchGroup/CROZtest/"
print os.listdir(folder)
allfiles  = [s for s in os.listdir(folder) if s.endswith('.tif')]
print allfiles
#filename = "/gpfs/projects/LynchGroup/CROZtest/WV03_20151024193848_1040010013779B00_15OCT24193848-P1BS-500656046010_01_P001_u08rf3031.tif"
for f in allfiles:
    filename= folder+f
    print filename 
   
    im3 = np.zeros((8000,8000),'uint8')
    
    with rasterio.open(filename) as src:
        print(src.width, src.height)
        r = src.read()
    if max(src.width,src.height)<40000:
        continue
    savedir = './im2/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    t =2000
    for i in range(0,3):
        for j in range(0,3):
            im = r[0][i*10000:(i+1)*10000,j*10000:(j+1)*10000]
            im2 = Image.fromarray(im)
            im2 = resizeimage.resize_cover(im2,[2000,2000])
            im3[i*t:(i+1)*t,j*t:(j+1)*t] = np.asarray(im2)
            im2.save(savedir+'/im_'+str(i)+str(j)+'.png')
    Image.fromarray(im3).save(f)
