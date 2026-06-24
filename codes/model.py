import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# MultiShapeKernel and Related Modules
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)
        

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.)
        return out_low + out_high




class Network(nn.Module):
    def __init__(self, 
                 input_vars, 
                 ex_vars, 
                 device, 
                 hyper_parameters,
                 max_x3_len=19,
                 unrestricted=True):
        super(Network, self).__init__()     
        self.input_vars = input_vars
        self.ex_vars = ex_vars
        self.num_fc_layers = hyper_parameters['fc_layers']
        self.fc_units = hyper_parameters['fc_units']
        self.r_units = hyper_parameters['r_units']
        self.bs = hyper_parameters['batch_size']
        self.dropout_rate = hyper_parameters['dropout']
        self.max_x3_len = max_x3_len
        self.unrestricted = unrestricted
        self.device = device
        
        # Replace LSTM with GRU
        self.gru1 = nn.GRU(self.input_vars, 
                           self.r_units, 
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)            
        
        self.map_h = nn.Linear((2 * self.r_units), self.r_units)
        
        self.gru2 = nn.GRU(self.input_vars - 1,
                           self.r_units,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        
        self.fc_layers = nn.ModuleList()
        if self.unrestricted:
            if self.num_fc_layers == 1:
                self.fc_layers.append(nn.Linear((self.r_units * 3) + self.ex_vars, 1))
            else:
                self.fc_layers.append(nn.Linear((self.r_units * 3) + self.ex_vars, self.fc_units))
        else:
            if self.num_fc_layers == 1:
                self.fc_layers.append(nn.Linear((self.r_units * 2) + self.ex_vars, 1))
            else:
                self.fc_layers.append(nn.Linear((self.r_units * 2) + self.ex_vars, self.fc_units))
            
        if self.num_fc_layers > 2:
            for i in range(2, self.num_fc_layers):
                self.fc_layers.append(nn.Linear(self.fc_units, self.fc_units))
            
        if self.num_fc_layers >= 2:
            self.fc_layers.append(nn.Linear(self.fc_units, 1))
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Only initialize h_0 for GRU
        self.h_0 = torch.zeros((2, self.bs, self.r_units)).to(self.device)
        
        self.ofl = 16
        self.conv1 = nn.Conv2d(1, self.ofl, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(3, self.r_units*2, kernel_size=1, padding='same')
        
        self.dyn1 = dynamic_filter(self.ofl)
        self.dyn2 = dynamic_filter(self.ofl)
        
        
        
    def forward(self, x, x2, x3=None):
        samples = x.shape[0]
        
 
        
        if samples < self.bs:
            x = F.pad(input=x, pad=(0, 0, 0, 0, 0, self.bs - samples), mode='constant', value=0)
            x2 = F.pad(input=x2, pad=(0, 0, 0, self.bs - samples), mode='constant', value=0)
            if x3 is not None:
                x3 = F.pad(input=x3, pad=(0, 0, 0, 0, 0, self.bs - samples), mode='constant', value=0)
                
        self.h_0 = self.h_0.data
        
        xc = x.clone().unsqueeze(-1)
        xc1 =  self.conv2 (self.conv1 (xc.permute(0,3,1,2)).permute(0,3,2,1)).permute(0,3,2,1)
        
        
        
        #xd1 = self.dyn1 ( xc1) 
        
        xd1 = self.dyn1 (self.dyn2 (xc1.permute(0,1,3,2)).permute(0,1,3,2))
        
        #xdd1 = self.dyn2 ( xd1) +xd2
        #xdd2 = self.dyn1 ( xd2) +xd1
        
        dxd = self.relu (torch.mean(xd1 ,1) )
        
        # GRU1 forward pass
        gru1_out, self.h_0 = self.gru1(x, self.h_0)
        gru1_out = self.dropout(gru1_out) + dxd
        
        
        #print(gru1_out.shape, self.r_units, self.ex_vars, xc1.shape, xd1.shape, xd2.shape) 
        
        #sys.exit()
        
        
        
        if x3 is not None:
            h2_0 = self.h_0.permute(1, 0, 2).flatten(1)
            h2_0 = self.tanh(self.map_h(h2_0))
            h2_0 = h2_0.reshape((self.bs, 1, self.r_units)).permute(1, 0, 2).data.contiguous()
            
            sequence_lengths = (x2[:, 1] * self.max_x3_len)
            sequence_lengths = sequence_lengths.type(torch.long).data
            
            batch_indices = [i for i in range(self.bs)]
            time_indices = [l for l in sequence_lengths]

            # GRU2 forward pass
            gru2_out, _ = self.gru2(x3, h2_0)
            gru2_out = gru2_out[batch_indices, time_indices, :].unsqueeze(1)
            gru2_out = self.dropout(gru2_out)
        
        if x3 is not None:
            gru_out = torch.cat((gru1_out[:, -1, :], gru2_out[:, -1, :]), dim=1)
        else:
            gru_out = gru1_out[:, -1, :]    
            
        x_ = torch.cat((gru_out, x2), dim=1)
        
        for i in range(len(self.fc_layers) - 1):
            x_ = self.fc_layers[i](x_)
            x_ = self.relu(x_)
            x_ = self.dropout(x_)
            
        x_ = self.fc_layers[len(self.fc_layers) - 1](x_)
        
        return x_[:samples, :]
