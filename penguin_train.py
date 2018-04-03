from models.models import create_model
from data.png_dataset import PngDataset
import numpy as np
from PIL import Image 
from options.train_options import TrainOptions
import time
from data.data_loader import CreateDataLoader


from util.visualizer import Visualizer

opt = TrainOptions().parse()

visualizer = Visualizer(opt)
total_steps = 0
data_loader = CreateDataLoader(opt)
#dataset = TifDataset(opt)
dataset = data_loader.load_data()
#print 'dataset size: ' + str(len(dataset))



#dataset.dprint()

model = create_model(opt)

for epoch in range(opt.epoch_count,opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    epoch_iter=0
    
    print('epoch:%d'%(epoch))

    for i,data in enumerate(dataset):
        model.set_input(data)
        model.optimize_parameters()
        if i% 5 ==0:
            visualizer.display_current_results(model.get_current_visuals(), epoch, False)
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, errors,10)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    if epoch % 5 ==0:
        model.save(epoch)
    model.update_learning_rate()
