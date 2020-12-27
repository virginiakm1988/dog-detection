import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch as to
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings
import os
import sys
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import glob
import os
import torch
from PIL import Image
import warnings
warnings.simplefilter("ignore", UserWarning)

from model import Siamese
from dataset import SiamDataset

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class ContrastiveLoss(nn.Module):
 
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        loss_contrastive = to.mean((1-label) * to.pow(euclidean_distance, 2) +
                                      (label) * to.pow(to.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

siamdset = SiamDataset(dir = input_dir)
batch_size = 16
train_dataloader = DataLoader(siamdset, shuffle=True, batch_size=batch_size,
                        num_workers=0)

                        
testset = SiamDataset(dir = input_dir, mode= "testing")
test_dataloader = DataLoader(testset, shuffle=True, batch_size=batch_size,
                        num_workers=0)

siam = Siamese().cuda()

number_epochs = 500
Criterion = ContrastiveLoss()
Optimizer = to.optim.Adam(siam.parameters(),lr = 0.001 )

counter = []
loss_history = [] 
iteration_number= 0
min_loss = 1000
start = 0
siam.train()
for epoch in range(start,start+number_epochs):
    torch.cuda.empty_cache()
    total_loss = 0
    for data in train_dataloader:
        #print(data)
   
        img1, img2 , label1, img3, img4, label2,c1,c2 = data
    
        Optimizer.zero_grad()
        
        # here we obtain the positive pairs' loss as well as the negative pairs' loss
        
        output1,output2 = siam(img1.cuda(),img2.cuda())
        output3,output4 = siam(img3.cuda(),img4.cuda())
        
        loss_pos = Criterion(output1,output2,label1.cuda())
        loss_neg = Criterion(output3,output4,label2.cuda())
        
        # the total loss is then computed and back propagated
        
        loss_contrastive = loss_pos + loss_neg
        
        loss_contrastive.backward()
        total_loss += loss_contrastive.cpu()
        Optimizer.step()
    
    siam.eval()
    val_pos_loss=0
    val_neg_loss=0
    torch.cuda.empty_cache()
    val_loss = 0
    for data in test_dataloader:
        img1, img2 , label1, img3, img4, label2,c1,c2 = data
        output1,output2 = siam(img1.cuda(),img2.cuda())
        output3,output4 = siam(img3.cuda(),img4.cuda())
        loss_pos = Criterion(output1,output2,label1.cuda())
        loss_neg = Criterion(output3,output4,label2.cuda())

        val_pos_loss += loss_pos.cpu()
        val_neg_loss += loss_neg.cpu()
        val_loss += (loss_pos.cpu() + loss_neg.cpu())
    
    # printing the train-20 errors
    if val_pos_loss + val_neg_loss < min_loss:
        min_loss = val_neg_loss+val_pos_loss
        torch.save(siam.state_dict(), os.path.join(output_dir,"model.pkl"))
    print("Epoch number {}\n  Current loss {}  Val loss {} val pos loss {} val neg loss{}\n".format(epoch,total_loss/siamdset.__len__(), (val_pos_loss+val_neg_loss)/testset.__len__(), val_pos_loss/testset.__len__(), val_neg_loss/testset.__len__()))
    counter.append(epoch+100)
    loss_history.append(loss_contrastive.item())
    siam.train()


