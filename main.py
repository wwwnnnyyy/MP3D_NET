# %%
# name:weinanyu
# date:20230706

# %%
import os
import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np
from DataFunc import loaddata_func, test_func, shuffle_func
from train import *
from model import *
from valid import *
from test import *
os.getcwd()
os.chdir
import numpy as np

path_G = '/home/weinanyu/try/data23/new25/g.npy'
path_M = '/home/weinanyu/try/data23/new25/m.npy'
path_EM = '/home/weinanyu/try/data23/new25/EM2.npy'
path_WELL = '/home/weinanyu/try/data23/new25/well.npy'
# path_G = '/home/weinanyu/try/data3/alldata_g.npy'
# path_M = '/home/weinanyu/try/data3/alldata_m.npy'
path_Y = '/home/weinanyu/try/data23/new25/y.npy'


test_g='/home/weinanyu/try/data_test141/g.npy'
test_m='/home/weinanyu/try/data_test141/m.npy'
test_em='/home/weinanyu/try/data_test141/EM2.npy'
test_well='/home/weinanyu/try/data_test141/well.npy'
test_y = '/home/weinanyu/try/data_test141/y.npy'
dataset = loaddata_func(path_G,path_M,path_EM,path_WELL,path_Y,number =200)
train_set,valid_set = shuffle_func(dataset,shuffle_ratio = 0.9)
test_set = test_func(test_g,test_m,test_em,test_well,test_y,number = 1)
# test_set = test_func(path_G,path_M,path_EM,path_WELL,path_Y,number =7)
# %%
from torch.utils.data import DataLoader
batch_size = 32
dataloader1 = DataLoader(train_set, batch_size,shuffle=True) # 训练集
dataloader3 = DataLoader(valid_set, batch_size,shuffle=True) # 验证集
dataloader5 = DataLoader(test_set, batch_size,shuffle=True) # 测试集
# %%

net =  Unet1()
print(net)
# %%

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net.load_state_dict(torch.load('best_model77.pth'))
#     device = torch.device('cpu')
    net.to(device=device)
    
    # train_net(net, device, dataloader1, dataloader3,dataloader3,epochs=2, batch_size = 128, lr=0.00001)
    net.load_state_dict(torch.load('best_model90.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    net.to(device=device)
    # train_net(net, device, dataloader1, dataloader3,dataloader3,epochs=400, batch_size = 256, lr=0.00001)
    plotpre(net,dataloader5,device)