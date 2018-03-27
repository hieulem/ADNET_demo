import numpy as np
import cv2
from scipy import misc
import os 
from m_util import sdmkdir,to_rgb3b
from sklearn import metrics
from vis import *
root = '/gpfs/projects/LynchGroup/Train_all/CROZtrain/CROPPED/p1000/' 
#root ='/gpfs/projects/LynchGroup/Train_all/CROPPED/p1000/'
root ='/nfs/bigbox/hieule/p1000/testing/PAUL/'
#root ='/nfs/bigbox/hieule/p1000/testing/CROZ/'
root='/nfs/bigbox/hieule/penguin_data/CROPPED/p300/'
visABC(root,'train_on_p300')
visABC(root,'train_on_p300_2')
visABC(root,'train_on_p300_3')

