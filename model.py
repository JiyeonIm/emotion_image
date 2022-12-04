import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Function, Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import copy 
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2 


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        
    @staticmethod
    def forward(self, input):
        '''
        input shape: (batch_size, num_classes*num_maps, w, h)
        '''
        num_maps = 4
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / num_maps)
        # 1) expand or divide dimension into 5 dimension
        x = input.view(batch_size, num_outputs, num_maps, h, w) # (batch_size, num_classes, num_maps, w, h)
        # average pooling
        # 2) sum over 2nd dimension
        output = torch.sum(x, 2) # (batch_size, num_classes, w, h)
        self.save_for_backward(input)
        # 3) divide by the number of 2nd dimension
        output = output.view(batch_size, num_outputs, h, w) / num_maps
        return output
    
    @staticmethod
    def backward(self, grad_output):
        num_maps = 4
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps).apply(input)
    
class ClassWisePoolFunction2(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction2, self).__init__()
        
    @staticmethod
    def forward(self, input):
        '''
        input shape: (batch_size, num_classes*num_maps, w, h)
        '''
        num_maps = 8
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / num_maps)
        # 1) expand or divide dimension into 5 dimension
        x = input.view(batch_size, num_outputs, num_maps, h, w) # (batch_size, num_classes, num_maps, w, h)
        # average pooling
        # 2) sum over 2nd dimension
        output = torch.sum(x, 2) # (batch_size, num_classes, w, h)
        self.save_for_backward(input)
        # 3) divide by the number of 2nd dimension
        output = output.view(batch_size, num_outputs, h, w) / num_maps
        return output
    
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        num_maps = 8
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)

class ClassWisePool2(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool2, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction2(self.num_maps).apply(input)

class ResNetWSL(nn.Module):
    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_ftrs = model.fc.in_features

        self.downconv = nn.Sequential(
            nn.Conv2d(2048, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.GAP = nn.AvgPool2d(14)
        self.GMP = nn.MaxPool2d(14)
        self.spatial_pooling = pooling # num feature detection k
        self.spatial_pooling2 = pooling2 # num classes c
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
            )

    def forward(self, x):
        x = self.features(x) # F = (b, n, h, w)
        x_ori = x # F = (b, n, h, w)
        # detect branch
        x = self.downconv(x) # F' = (b, kc, h, w)
        f_2 = x # F' = (b, kc, h, w)             
        x = self.GAP(x)  #x = self.GMP(x) # v_c = (b, kc, 1, 1)
        x = self.spatial_pooling(x) # v = (b, c, 1, 1)
        v = x.view(x.size(0), -1) # v = (b, c)
        
        # cls branch
        m_c = self.spatial_pooling(f_2) # (b, c, h, w) 각 클래스별로 k개의 map을 avg pooling
        
        weighted_m_c = m_c * v.view(v.size(0),v.size(1),1,1) # generating sentiment map M (weighted sum)
        m = self.spatial_pooling2(weighted_m_c) # M = (b, 1, h, w)
        x_conv_copy = m # M = (b, 1, h, w)
        
        for num in range(0,2047): # n     
            x_conv_copy = torch.cat((x_conv_copy, m),1)  # extended M = (b, n, h, w)
            
        u = torch.mul(x_conv_copy,x_ori) # U = (b, n, h, w)
        f_u = torch.cat((x_ori,u),1) # F + U = (b, 2n, h, w)
        d = self.GAP(f_u) # d = (b, 2n, 1, 1)
        d = d.view(d.size(0),-1) # d = (b, 2n)
        output = self.classifier(d) # (b, c)
        return v, output, m, m_c

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling2 = nn.Sequential()
        pooling2.add_module('class_wise', ClassWisePool2(num_classes))
        model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)
        input_size = 448

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size