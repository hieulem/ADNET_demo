import numpy as np
import torch
import os
from base_model import BaseModel
import networks
from torch.autograd  import Variable

class Unet256Model(BaseModel):
    net = None
    gpu = []
    def name(self):
        return 'Unet256'
    def __init__(self, gpu_ids=[0],load_model=None):
        self.gpu = gpu_ids
        norm_layer = networks.get_norm_layer('instance')
        self.net =networks.UnetGenerator(3,1,8,64,use_dropout=False,norm_layer = norm_layer,gpu_ids = gpu_ids)
        if load_model is not None:
            print "loading model"
            self.net.load_state_dict(torch.load(load_model))
        self.net.cuda(0)
    def print_net(self):
        networks.print_network(self.net)
    def test(self,data):
        return self.net.forward(Variable(data['A'].cuda(0),requires_grad = 0)).data
    
