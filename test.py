from torch import optim
import torch.nn as nn
import torch
import os
import numpy as np
# %matplotlib notebook
import matplotlib.pyplot as plt
from model import *

def iou(pred, target, n_classes = 2):
#n_classes ：the number of classes in your dataset
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
        return np.array(ious)

def plotpre(model,loader,device):
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    total = len(loader.dataset)
    for x,x1,x2,x3,y in loader:
        x,y = x.to(device),y.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        with torch.no_grad():
            pred = model(x,x1,x2,x3) 
        pred0 = torch.as_tensor((pred - 0.15) > 0, dtype=torch.int64) 
        pred1 = pred.detach().numpy()
        pred00 = pred0.detach().numpy()
        
        y1 = y.numpy()
#         print(pred1.size)
        acc = iou(pred0, y)
        print(acc)
        pred01 = pred00.reshape(1,20,20,20)
        pred2 = pred1.reshape(1,20,20,20)
        y2 = y1.reshape(1,20,20,20)
        print(y2.shape)
        fig = plt.figure()
        
        ax1 = plt.subplot(3,3,1)   
        h1 = plt.imshow(np.transpose(y2[0,9,:,:]))
        c1 = plt.colorbar(h1)
        plt.xticks([])
        plt.yticks([])
        
        
        ax2 = plt.subplot(3,3,2) 
        h2 = plt.imshow(np.transpose(pred2[0,9,:,:]))
        plt.colorbar(h2)
        
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
        ax3 = plt.subplot(3,3,3) 
        h22 = plt.imshow(np.transpose(pred01[0,9,:,:]))
        plt.colorbar(h22)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        
        
        
        
        ax4 = plt.subplot(3,3,4)   
        h3 = plt.imshow(np.transpose(y2[0,:,9,:]))
        c3 = plt.colorbar(h3)
        plt.xticks([])
        plt.yticks([])
        

        
        ax5 = plt.subplot(3,3,5) 
        h4 = plt.imshow(np.transpose(pred2[0,:,9,:]))
        plt.colorbar(h4)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        
        ax6 = plt.subplot(3,3,6) 
        h44 = plt.imshow(np.transpose(pred01[0,:,9,:]))
        plt.colorbar(h44)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        
        
        ax7 = plt.subplot(3,3,7)   
        h5 = plt.imshow(y2[0,:,:,9])
        c5 = plt.colorbar(h5)
        plt.xticks([])
        plt.yticks([])
        
        
        ax8 = plt.subplot(3,3,8) 
        h6 = plt.imshow(pred2[0,:,:,9])
        plt.colorbar(h6)
        plt.show()
        plt.xticks([])
        plt.yticks([])
        
        ax9 = plt.subplot(3,3,9) 
        h66 = plt.imshow(pred01[0,:,:,9])

        plt.colorbar(h66)
        plt.xticks([])
        plt.yticks([])
       
        
        plt.show()
        print(pred0)
        np.savetxt('/home/weinanyu/try/data25/si.txt',pred0)

        
    return pred
