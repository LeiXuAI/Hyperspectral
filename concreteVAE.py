import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from utils import get_activation, MSELoss
import math


from encoders import ConcreteGates, ConcreteMask, ConcreteMax, ConcreteNew, ConcreteSelector

class ConcreteVAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim, 
                 selected_num, 
                 loss_rec=nn.CrossEntropyLoss(),
                 loss_com=MSELoss(),
                 lam1=0.005,
                 lam2=1,
                 encoder_type='concrete_gates',
                 **kwargs):
        super(ConcreteVAE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.selected_num = selected_num
        self.encoder_type = encoder_type
        self.loss_rec = loss_rec
        self.loss_com = loss_com
        self.lam1 = lam1
        self.decoder = Decoder(self.selected_num, self.hidden_dim, self.output_dim)
        if self.encoder_type == 'concrete_selector':
            # for future exploration
            self.lam1 = 0
            self.sampled_layer = ConcreteSelector(self.input_dim, self.selected_num)
        elif self.encoder_type == 'concrete_mask':
            # for future exploration
            self.lam1 = 0
            self.sampled_layer = ConcreteMask(self.input_dim, self.output_dim, self.selected_num)
        elif self.encoder_type == 'concrete_gates':
            #This work only uses concrete_gates.
            self.lam1 = 0.005
            self.sampled_layer = ConcreteGates(self.input_dim, self.output_dim, self.selected_num)
        elif self.encoder_type == 'concrete_max':
            # for future exploration
            self.lam1 = 0
            self.sampled_layer = ConcreteMax(self.input_dim, self.selected_num)
        elif self.encoder_type == 'concrete_new':
            # for future exploration
            self.lam1 = 0.00
            self.sampled_layer = ConcreteNew(self.input_dim, self.selected_num)
    
    def forward(self, x, **kwargs):
        loss_penalty = 0
        if self.lam1:
            sampled_x, m = self.sampled_layer(x)
            loss_penalty = torch.mean(torch.sum(m, dim=-1))
    
        _, m = self.sampled_layer(x)
        selected_inds = self.sampled_layer.get_inds(num_features=self.selected_num)
    
        selected_x = x[:, selected_inds]
        y_select = self.decoder(selected_x)
        loss_rec = self.loss_rec(x, y_select)
        loss = loss_rec + self.lam1*loss_penalty 
        return selected_x, selected_inds, loss, loss_penalty

class Decoder(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                output_dim,
                activation='relu',
                batch_norm=False,
                **kwargs):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
        fc_layers = [nn.Linear(input_dim, output_dim) for input_dim, output_dim in
                              zip([self.input_dim] + self.hidden_dim, self.hidden_dim + [self.output_dim])]
        self.fc_layers = nn.ModuleList(fc_layers)
        self.activation = get_activation(activation)
        if batch_norm:
            normal_layers = [nn.BatchNorm1d(d) for d in self.hidden_dim]
        else:
            normal_layers = [nn.Identity() for d in self.hidden_dim]
        self.normal_layers = nn.ModuleList(normal_layers)

    def forward(self, x):
        for fc, norm in zip(self.fc_layers, self.normal_layers):
            x = fc(x)
            x = self.activation(x)
            x = norm(x) 
        return torch.sigmoid(self.fc_layers[-1](x))

    

        

