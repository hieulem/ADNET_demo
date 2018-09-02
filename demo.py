from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image 
dataset = SingleDataset('../datasets/SBUsd/Test/TestA/')
print('dataset size: ' + str(len(dataset)))

model = Unet256Model(load_model='135_net_D.pth')
model.print_net()

for i,data in enumerate(dataset):
    out = model.test(data)
    im_out = out[0].cpu().float().numpy()
    im_out = np.transpose(im_out,(1,2,0))
    im_out = (im_out+1)/2*255
    im_out = im_out.astype('uint8')
    
    A = Image.fromarray(np.squeeze(im_out,axis =2)).resize((data['w'],data['h']))
    A.save('out/'+data['imname'])
