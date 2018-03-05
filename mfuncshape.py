
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

def BoundingBoxShape(bb,padding=0):
    #bb = [minx,miny,maxx,maxy]
    bb_shape = []
    c = []
    bb[0] = bb[0] - padding
    bb[1] = bb[1] - padding
    bb[2] = bb[2] + padding
    bb[3] = bb[3] + padding
    c.append([bb[0],bb[1]])
    c.append([bb[0],bb[3]])
    c.append([bb[2],bb[3]])
    c.append([bb[2],bb[1]])
    c.append([bb[0],bb[1]])

    a1 = Polygon(c)
    bb_shape.append({'geometry':mapping(a1),'properties':{'id':123}})
    return bb_shape
        
def ReadProjection(filename):
    prj_file = open(filename)
    shape_prj = osr.SpatialReference()
    k = prj_file.read()
    shape_prj.ImportFromWkt(k)
    return shape_prj

def TransformShape(oshape,coordTrans,padding = 0):
    newshape = []
    maxx = -99999999
    maxy = -99999999
    minx = 99999999
    miny = 99999999
    for piece  in oshape:
        geo = shape(piece['geometry'])
        c=[]
        for x,y in geo.exterior.coords:
            x,y,z = coordTrans.TransformPoint(x,y)
            c.append([x,y])
            maxx= max(maxx,x)
            maxy=max(maxy,y)
            minx=min(minx,x)
            miny=min(miny,y)
        sp = Polygon(c)
        newshape.append({
            'geometry': mapping(sp),
            'properties': {'id':123},
        })
    bb = [minx,miny,maxx,maxy]


    bbshape = BoundingBoxShape(bb,padding)
    return newshape,bbshape

