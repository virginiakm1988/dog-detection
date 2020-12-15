import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch as to
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class Siamese(nn.Module):
    
    def __init__(self):
        super(Siamese,self).__init__()
        
        # A simple two layer convolution followed by three fully connected layers should do
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        
    def forward_once(self,x):
        out = self.cnn(x)
        #print(out.shape)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
    def forward(self, x, y):    
        
        # doing the forwarding twice so as to obtain the same functions as that of twin networks
        
        out1 = self.forward_once(x)
        out2 = self.forward_once(y)
        
        return out1, out2
    
    
    def evaluate(self, x, y):
        
        # this can be used later for evalutation
        
        m = to.tensor(1.0, dtype=to.float32).cuda()
        
        if type(m) != type(x):
            x = to.tensor(x, dtype = to.float32, requires_grad = False).cuda()
            
        if type(m) != type(y):
            y = to.tensor(y, dtype = to.float32, requires_grad = False).cuda()
        
        x = x.view(-1,3,128,128)
        y = y.view(-1,3,128,128)
        
        with to.no_grad():
            
            out1, out2 = self.forward(x, y)
            
            return nn.functional.pairwise_distance(out1, out2)
        