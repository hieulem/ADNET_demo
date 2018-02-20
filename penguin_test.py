from models.models import create_model
from data.tif_dataset import TifDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util.visualizer import Visualizer
import time
from data.data_loader import CreateDataLoader
import matplotlib.pyplot as plt


opt = TestOptions().parse()
opt.dataroot = './CROPPED/p200/'+str(opt.i)+'/'
opt.dataset_mode='tif'
opt.fineSize = 256
opt.model ='single_unet'
opt.display_port = 9998
opt.resdir = './results/'
#data_loader = CreateDataLoader(opt)
dataset = TifDataset(opt)
#dataset = data_loader.load_data()
print 'dataset size: ' + str(len(dataset))

opt.which_epoch=115

#dataset.dprint()

model = create_model(opt)
visualizer = Visualizer(opt)
print dataset
nx,ny = dataset.get_number_of_patches()
allpred = np.zeros(((nx+1)*256,(ny+1)*256,3),dtype=np.uint8)
allinput = np.zeros(((nx+1)*256,(ny+1)*256,3),dtype=np.uint8)

allGT = np.zeros(((nx+1)*256,(ny+1)*256,3),dtype=np.uint8)

for i in range(0,nx):
    res =[]
    for j in range(0,ny):
        data = dataset.getpatch(i,j)
        model.set_input(data)
        out = model.get_prediction()
        print out['output'].shape
        
        allinput[i*256:(i+1)*256,j*256:(j+1)*256,:] = out['input']
        allpred[i*256:(i+1)*256,j*256:(j+1)*256,:] = out['output']
        allGT[i*256:(i+1)*256,j*256:(j+1)*256,:] = out['GT']
print data['imname']        
Image.fromarray(allpred).save(opt.resdir+data['imname']+'_out.png')
Image.fromarray(allinput).save(opt.resdir+data['imname']+'_input.png')
Image.fromarray(allGT).save(opt.resdir+data['imname']+'_GT.png')
