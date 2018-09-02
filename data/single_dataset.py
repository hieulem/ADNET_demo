import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
class SingleDataset(BaseDataset):
    def __init__(self, dataroot):
        self.root = dataroot
        self.dir_A = os.path.join(dataroot)

        self.A_paths,self.imname = make_dataset(self.dir_A)
        
        self.A_paths = sorted(self.A_paths)
        self.imname = sorted(self.imname)

    def __getitem__(self, index):
        
        A_path = self.A_paths[index]
        imname = self.imname[index]
        A_img = Image.open(A_path).convert('RGB')
        ow = A_img.size[0]
        oh = A_img.size[1]
        A_img = A_img.resize((256,256))
        A_img = np.array(A_img,np.float32)
        A_img = np.log(A_img +1)
        A_img = torch.from_numpy(A_img.transpose(2, 0, 1)).div(np.log(256))
        A_img = A_img-0.5
        A_img = A_img*2
        A = A_img.unsqueeze(0)
        return {'A': A, 'A_paths': A_path,'imname':imname,'w':ow,'h':oh}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
