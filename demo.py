from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image 
dataset = SingleDataset('Test')
print 'dataset size: ' + str(len(dataset))

model = Unet256Model(load_model='50_net_D.pth')
model.print_net()

for i,data in enumerate(dataset):
    print data['imname']
    out = model.test(data)
    im_out = out[0].cpu().float().numpy()
    im_out = np.transpose(im_out,(1,2,0))
    im_out = (im_out+1)/2*np.log(256)
    im_out = np.exp(im_out)-1
    im_out = im_out.astype('uint8')
    Image.fromarray(np.squeeze(im_out,axis =2)).save('out/'+data['imname'])

