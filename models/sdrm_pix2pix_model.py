import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn

class SDRMPix2PixModel(BaseModel):
    def name(self):
        return 'SDRMPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        # load/define networks
        self.netG = networks.define_G(4, 3, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.ce_loss = nn.CrossEntropyLoss()        
        #use_sigmoid = opt.no_lsgan
        
        self.netD = networks.define_G(3, 1, opt.ngf,
                                      opt.which_model_netD, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        #if not self.isTrain or opt.continue_train:
        self.load_network(self.netG, 'G', opt.which_epoch)
        self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1,0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr,betas=(opt.beta1,0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        #handle hyper parameters:
        if self.isTrain:
            self.opt.lambda_outside_mask = 50
            self.opt.lambda_GAN=1
            self.opt.lambda_pos=0.5
            self.opt.lambda_real=0.9
            self.opt.lambda_sd = 1
            self.opt.lambda_neg = 1-self.opt.lambda_pos
            self.opt.lambda_fake = 1-self.opt.lambda_real
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        self.netG.cuda()
        self.netD.cuda()
        networks.print_network(self.netD)
        print('-----------------------------------------------')
    global L2Loss
    global L1Loss
    def L2Loss(A,B):
        if A.dim()>0:
            return (A - B).pow(2).mean()
        else:
            return 0
    
    def L1Loss(A,B):
        if A.dim()>0:
            return (A-B).abs().mean()
        else:
            return 0
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_B = input_B[:,0:1,:,:] 
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
         
        self.shadow_mask = Variable(input_B.cuda())
        self.shadow_image = Variable(input_A.cuda())


    def forward(self):
        input_for_G = torch.cat((self.shadow_image,self.shadow_mask),1)
        self.nonsdim = self.netG.forward(input_for_G.cuda())
            
    def backward_D_fake(self):

        self.pred_sd_fake = self.netD.forward(self.nonsdim.detach())

        #loss on sd pixel:
        pos_ind = self.shadow_mask!=0
        neg_ind = 1 + (-1)* pos_ind
        
        self.loss_D_sd_fake_pos = L2Loss(self.pred_sd_fake[pos_ind],self.shadow_mask[pos_ind]) * self.opt.lambda_pos
        self.loss_D_sd_fake_neg = L2Loss(self.pred_sd_fake[neg_ind],self.shadow_mask[neg_ind]) * self.opt.lambda_neg 
            
        
        self.loss_D_sd_fake = self.loss_D_sd_fake_pos + self.loss_D_sd_fake_neg
        self.loss_D_fake = self.loss_D_sd_fake
        
        self.loss_D_fake.backward()
        self.loss_D_pos = self.loss_D_sd_fake_pos
        self.loss_D_neg = self.loss_D_sd_fake_neg




    def backward_D(self):
        #get prediction:
        shadow_mask_3D = self.shadow_mask.repeat(1,3,1,1)
        shadow_mask_3D[shadow_mask_3D>0] = 1
        shadow_mask_3D[shadow_mask_3D<0] =0
        self.im_to_D= self.nonsdim.detach().mul(shadow_mask_3D) +  self.shadow_image.detach().mul(1-shadow_mask_3D.detach())
        self.pred_sd_fake = self.netD.forward(self.im_to_D)
        self.pred_sd_real = self.netD.forward(self.shadow_image)

        
        
        
        #loss on sd pixel:
        pos_ind = self.shadow_mask!=0
        neg_ind = 1 + (-1)* pos_ind

        self.loss_D_sd_real_pos = L2Loss(self.pred_sd_real[pos_ind],self.shadow_mask[pos_ind])* self.opt.lambda_pos * self.opt.lambda_real
        self.loss_D_sd_fake_pos = L2Loss(self.pred_sd_fake[pos_ind],self.shadow_mask[pos_ind])* self.opt.lambda_pos * self.opt.lambda_fake
        self.loss_D_sd_real_neg = L2Loss(self.pred_sd_real[neg_ind],self.shadow_mask[neg_ind])* self.opt.lambda_neg * self.opt.lambda_real
        self.loss_D_sd_fake_neg = L2Loss(self.pred_sd_fake[neg_ind],self.shadow_mask[neg_ind])* self.opt.lambda_neg * self.opt.lambda_fake
        self.loss_D_sd_real = self.loss_D_sd_real_pos + self.loss_D_sd_real_neg
        self.loss_D_sd_fake = self.loss_D_sd_fake_pos + self.loss_D_sd_fake_neg
        self.loss_D_sd = (self.loss_D_sd_real + self.loss_D_sd_fake) * self.opt.lambda_sd
        self.loss_D = self.loss_D_sd

        #for visualization only:
        self.loss_D_real = self.loss_D_sd_real
        self.loss_D_fake = self.loss_D_sd_fake
        self.loss_D_pos = self.loss_D_sd_real_pos + self.loss_D_sd_fake_pos
        self.loss_D_neg = self.loss_D_sd_real_neg + self.loss_D_sd_fake_neg
        #done dummy code for visualization

        #loss on boundary:

        self.loss_D.backward()
    
    def backward_G(self):
        
        #Get the prediction:
        
        shadow_mask_3D = self.shadow_mask.repeat(1,3,1,1)
        shadow_mask_3D[shadow_mask_3D>0] =1
        shadow_mask_3D[shadow_mask_3D<0] =0

        update_part = self.nonsdim.mul(shadow_mask_3D)
        #if self.opt.not_care_outside:
        nonupdate_part = self.shadow_image.detach().mul(1-shadow_mask_3D.detach())
        #else:
        #    nonupdate_part = self.nonsdim.detach().mul(1-shadow_mask_3D.detach())
                
        self.im_to_D = update_part + nonupdate_part

        self.pred_fake_sd = self.netD.forward(self.im_to_D)
        
        self.loss_G_L1 = 0
        #loss GAN on the shadow mask:
        
        select_sd_region = self.pred_fake_sd[self.shadow_mask!=0]
        if select_sd_region.dim()>0:
            self.loss_G_GAN = select_sd_region.pow(2).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN

                
        
        

        
        self.loss_G.backward()
        del shadow_mask_3D 
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):

        A = OrderedDict()
        A['G_GAN'] = self.loss_G_GAN.data[0]
        A['D_pos'] = self.loss_D_pos.data[0]
        
       
        return A

    def get_current_visuals(self):
        
        
        A = OrderedDict()
        scl =255
        A['ori'] = util.tensor2im(self.shadow_image.data)
        if hasattr(self,'shadow_mask'):
            A['sd_mask'] = util.tensor2im(self.shadow_mask.data) 
        
        if hasattr(self,'nonsdim'):
            A['nonsd_pred'] = util.tensor2im(self.nonsdim.data)
        
        if hasattr(self,'pred_sd'):
            A['pred_sd'] = util.tensor2im(self.pred_sd.data)
            
        if hasattr(self,'pred_sd_real'):
            A['pred_sd_real'] = util.tensor2im(self.pred_sd_real.data)         
        if hasattr(self,'im_to_D'):
            A['im_to_D'] = util.tensor2im(self.im_to_D.data)         
        if hasattr(self,'pred_sd_fake'):
            A['pred_sd_fake'] = util.tensor2im(self.pred_sd_fake.data)         
        return A
    
    def get_prediction(self):
        input_for_G = torch.cat((self.shadow_image,self.shadow_mask),1)
        self.nonsdim = self.netG.forward(input_for_G)
        shadow_mask_3D = self.shadow_mask.repeat(1,3,1,1)
        shadow_mask_3D[shadow_mask_3D>0] = 1
        shadow_mask_3D[shadow_mask_3D<0] =0
        self.im_to_D= self.nonsdim.detach().mul(shadow_mask_3D) +  self.shadow_image.detach().mul(1-shadow_mask_3D.detach())
        self.pred_sd = self.netD.forward(self.shadow_image)
        A = OrderedDict()
        A['im_to_D'] =  util.tensor2im(self.im_to_D.data)
        A['pred_sd'] = util.tensor2im(self.pred_sd.data)
        return A    
    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
