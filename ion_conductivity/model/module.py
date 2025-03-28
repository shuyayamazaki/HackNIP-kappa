import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
from typing import Optional
from model.graph_encoder import *
from model.utils import *


class BaseModule(pl.LightningModule):
    '''
    Base module for all models.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = cfg.optimization.lr

    def configure_optimizers(self):
        # optimizer
        if self.cfg.optimization.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.cfg.optimization.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.cfg.optimization.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler
        if self.cfg.optimization.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.cfg.optimization.scheduler_params)
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}
        elif self.cfg.optimization.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.cfg.optimization.scheduler_params)
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}
        elif self.cfg.optimization.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.cfg.optimization.scheduler_params)
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': self.cfg.optimization.monitor}


class IonConductivityCGCNNModule(BaseModule):
    '''
    AutoEncoder model take (batch_size, 1024, 5) img as input and return (batch_size, 1024, 5) img as output
    CNN model is used for encoder and decoder.
    Latent vector size is 128.    
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.graph_encoder = CGCNN(**cfg.model.cgcnn)
        self.temperature_encoder = build_mlp(**cfg.model.temperature_encoder)
        self.readout = build_mlp(**cfg.model.readout)

    def forward(self, data):
        h_graph = self.graph_encoder(data)
        h_temp = self.temperature_encoder(data.temperature.unsqueeze(1))
        out = self.readout(torch.concat([h_graph, h_temp], dim=1))
        return out
        
    def loss(self, label, pred, mode='train'):
        if mode == 'train':
            loss = F.huber_loss(pred.squeeze(), 100*10**label)
        else:
            loss = F.l1_loss(pred.squeeze(), 100*10**label)
        self.log(f'{mode}_loss', loss, prog_bar=True, batch_size=len(label), sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='val')
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='test')
        return loss


class IonConductivityMACEModule(BaseModule):
    '''
    AutoEncoder model take (batch_size, 1024, 5) img as input and return (batch_size, 1024, 5) img as output
    CNN model is used for encoder and decoder.
    Latent vector size is 128.    
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.graph_encoder = MACE(**cfg.model.cgcnn)
        self.temperature_encoder = build_mlp(**cfg.model.temperature_encoder)
        self.readout = build_mlp(**cfg.model.readout)

    def forward(self, data):
        h_graph = self.graph_encoder(data)
        h_temp = self.temperature_encoder(data.temperature.unsqueeze(1))
        out = self.readout(torch.concat([h_graph, h_temp], dim=1))
        return out
        
    def loss(self, label, pred, mode='train'):
        if mode == 'train':
            loss = F.huber_loss(pred.squeeze(), 100*10**label)
        else:
            loss = F.l1_loss(pred.squeeze(), 100*10**label)
        self.log(f'{mode}_loss', loss, prog_bar=True, batch_size=len(label), sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='val')
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch.y, out, mode='test')
        return loss

