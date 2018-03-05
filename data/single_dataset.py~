import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class SingleDataset(BaseDataset):
    def __init__(self, dataroot):
        self.root = dataroot
        self.dir_A = os.path.join(dataroot)

        self.A_paths,self.imname = make_dataset(self.dir_A)
        
        self.A_paths = sorted(self.A_paths)
        self.imname = sorted(self.imname)
        transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),
                                                 (0.5,0.5,0.5))]

        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        
        A_path = self.A_paths[index]
        imname = self.imname[index]
        print imname
        A_img = Image.open(A_path).convert('RGB')

        print "resize to 256x256"
        A_img = A_img.resize((256,256),Image.NEAREST)
        A_img = np.log(np.asarray(A_img).astype(np.double) +1)
        A_img = A_img/np.log(256) *255

        A = self.transforms(A_img)
        A = A.unsqueeze(0)
        return {'A': A, 'A_paths': A_path,'imname':imname}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
