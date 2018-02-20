from models.models import create_model
from data.tif_dataset import TifDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import time
from data.data_loader import CreateDataLoader

opt = TrainOptions().parse()
opt.dataset_mode='tif'
opt.dataroot = './CROPPED/p200'
opt.fineSize = 256
opt.model ='single_unet'
opt.display_port = 9998

data_loader = CreateDataLoader(opt)
#dataset = TifDataset(opt)
dataset = data_loader.load_data()
print 'dataset size: ' + str(len(dataset))



#dataset.dprint()

model = create_model(opt)
visualizer = Visualizer(opt)

for epoch in range(opt.epoch_count,opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    epoch_iter=0
    for i,data in enumerate(dataset):
        print('epoch:%d, i: %d'%(epoch,i))
        model.set_input(data)
        model.optimize_parameters()
        if i % 30 ==0:
            visualizer.display_current_results(model.get_current_visuals(),epoch,True)
    
    print('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    if epoch % 5 ==0:
        model.save(epoch)
    model.update_learning_rate()
