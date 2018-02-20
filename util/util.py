from __future__ import print_function
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#import Image
import inspect, re
import numpy as np
import os
import collections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im_logc(image_tensor, imtype=np.uint8,scale=255):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
#    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    image_numpy = (image_numpy+1) /2 
    image_numpy = image_numpy * (np.log(scale+1)) 
   
    image_numpy = np.exp(image_numpy) -1
    if scale == 1:
        image_numpy = image_numpy * 255
    return image_numpy.astype(imtype)

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    if image_numpy.shape[2] == 8:
        a = np.tile(np.expand_dims(np.mean(image_numpy,axis=2),axis=2),(1,1,3))
        image_numpy = Image.fromarray(a.astype(np.uint8))
        #image_numpy = Image.fromarray(np.uint8(image_numpy[:,:,0:3]))
    elif image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, (1,1,3))
    image_numpy  = np.asarray(image_numpy)
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
