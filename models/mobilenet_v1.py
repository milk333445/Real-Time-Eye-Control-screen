import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), # input:224x224x3 out:(224-3+2*1)/1+1=224
        nn.BatchNorm2d(in_channels), 
        nn.ReLU6(inplace=True), 
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), 
        nn.BatchNorm2d(out_channels), 
        nn.ReLU6(inplace=True) 
    )
    
class MobileNetV1(nn.Module): # parameter: 3208001
    def __init__(self, num_classes=1, pooling_type='avg'):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), # input:(1, 3, 224, 224) out:(1, 32, 112, 112)
            nn.BatchNorm2d(32), 
            nn.ReLU6(inplace=True),
            conv_dw(32, 64, 1), # input:(1, 32, 112, 112) out:(1, 64, 112, 112)
            conv_dw(64, 128, 2), # input:(1, 64, 112, 112) out:(1, 128, 56, 56)
            conv_dw(128, 128, 1), # input:(1, 128, 56, 56) out:(1, 128, 56, 56)
            conv_dw(128, 256, 2), # input:(1, 128, 56, 56) out:(1, 256, 28, 28)
            conv_dw(256, 256, 1), # input:(1, 256, 28, 28) out:(1, 256, 28, 28)
            conv_dw(256, 512, 2), # input:(1, 256, 28, 28) out:(1, 512, 14, 14)
            conv_dw(512, 512, 1), # input:(1, 512, 14, 14) out:(1, 512, 14, 14)
            conv_dw(512, 512, 1), # input:(1, 512, 14, 14) out:(1, 512, 14, 14)
            conv_dw(512, 512, 1), # input:(1, 512, 14, 14) out:(1, 512, 14, 14)
            conv_dw(512, 512, 1), # input:(1, 512, 14, 14) out:(1, 512, 14, 14)
            conv_dw(512, 512, 1), # input:(1, 512, 14, 14) out:(1, 512, 14, 14)
            conv_dw(512, 1024, 2), # input:(1, 512, 14, 14) out:(1, 1024, 7, 7)
            conv_dw(1024, 1024, 1), # input:(1, 1024, 7, 7) out:(1, 1024, 7, 7)
        )    
        
        if pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(1024, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.model(x) # (1, 1024, 7, 7)
        x = self.pool(x) #(1, 1024, 1, 1)
        x = x.view(-1, 1024) #(1, 1024)
        x = self.fc(x) # (1, 1)
        return x
    
#model = MobileNetV1()
#x = torch.randn(1, 3, 712, 712)
#print(model(x).shape) # torch.Size([1, 1])
