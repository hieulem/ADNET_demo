from models.models import create_model
from data.tif_dataset import TifDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
import time
from data.data_loader import CreateDataLoader

opt = TrainOptions().parse()
opt.model ='single_unet'

data_loader = CreateDataLoader(opt)
#dataset = TifDataset(opt)
dataset = data_loader.load_data()
#print 'dataset size: ' + str(len(dataset))



#dataset.dprint()

model = create_model(opt)

for epoch in range(opt.epoch_count,opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    epoch_iter=0
    for i,data in enumerate(dataset):
        print('epoch:%d, i: %d'%(epoch,i))
        model.set_input(data)
        model.optimize_parameters()
    
    print('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    if epoch % 5 ==0:
        model.save(epoch)
    model.update_learning_rate()
