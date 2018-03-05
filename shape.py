
from shapely.geometry import shape,mapping
from shapely.geometry.polygon import LinearRing, Polygon
import fiona 
from PIL import Image 
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from descartes import PolygonPatch
from osgeo import gdal,osr
import pycrs

shapefile = '/gpfs/projects/LynchGroup/CROZtest/AnnotatedGuano/crozier_guanoarea_stereographic'
TIFfile = '/gpfs/projects/LynchGroup/CROZtest/WV03_20151024193848_1040010013779B00_15OCT24193848-P1BS-500656046010_01_P001_u08rf3031'
sf = shapefile+'.prj'
annofile = shapefile+'.shp'
tf = TIFfile + '.prj' 
TIFim = TIFfile + '.tif'
def read_prj_from_file(file):
    prj_file = open(file)
    shape_prj = osr.SpatialReference()
    k = prj_file.read() 
    shape_prj.ImportFromWkt(k)
    return shape_prj


shape_prj = read_prj_from_file(sf)
tif_prj = read_prj_from_file(tf)

dataset= rasterio.open(TIFim)


coordTrans = osr.CoordinateTransformation(shape_prj, tif_prj)
i=0
maxx = -99999999
maxy = -99999999
minx = 99999999
miny = 99999999

anno = fiona.open(annofile)
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}
with fiona.open('transformed_shape.shp', 'w', 'ESRI Shapefile', schema) as shfile:
    for obs in anno:
        shp_geom = shape(obs['geometry'])
        c =[]
        for x,y in shp_geom.exterior.coords:
            x,y,z =  coordTrans.TransformPoint(x,y)
            maxx= max(maxx,x)
            maxy=max(maxy,y)
            minx=min(minx,x)
            miny=min(miny,y)
            c.append([x,y])
        sp = Polygon(c)
        fig = plt.figure(1, figsize=(20,20), dpi=90)
        shfile.write({
            'geometry': mapping(sp),
            'properties': {'id': 123},
        })
        ax = fig.add_subplot(111)
        ring_patch = PolygonPatch(sp)
        ax.set_title('Annotated guano')
        ax.add_patch(ring_patch)
xrange = [minx, maxx]
yrange = [miny, maxy]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_aspect(1)

plt.hold(True)

plt.show()


