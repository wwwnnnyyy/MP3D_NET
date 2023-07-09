
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
def loaddata_func(path1,path2,path4,path5,path3,number):

    M_ori = np.load(path2)
    G_ori = np.load(path1)
    EM_ori = np.load(path4)
    WELL_ori = np.load(path5)
    Y_ori = np.load(path3)
    M_ori = M_ori[:200,:,:,:]
    G_ori = G_ori[:200,:,:,:]
    EM_ori = EM_ori[:200,:,:,:]
    WELL_ori = WELL_ori[:200,:,:]
    Y_ori = Y_ori[:200,:,:,:]

    train_M=M_ori.reshape([number,-1])
    train_M = train_M.reshape([number,1,10,10])
    train_G=G_ori.reshape([number,-1])
    train_G = train_G.reshape([number,1,20,20])
    train_EM=EM_ori.reshape([number,-1])
    train_EM = train_EM.reshape([number,2,10,5,8])


    train_WELL=WELL_ori.reshape([number,-1])
    train_WELL = train_WELL.reshape([number,2,20])

    train_Y=Y_ori.reshape([number,-1])
    train_Y = train_Y.reshape([number,8000])


    train_M=train_M.astype(np.float32)
    train_G=train_G.astype(np.float32)
    train_EM=train_EM.astype(np.float32)
    train_WELL=train_WELL.astype(np.float32)
    train_Y=train_Y.astype(np.float32)
    M = torch.from_numpy(train_M)
    G = torch.from_numpy(train_G)
    EM = torch.from_numpy(train_EM)
    WELL = torch.from_numpy(train_WELL)
    Y=torch.from_numpy(train_Y)
    
    # dataset_all=TensorDataset(M,G,EM,WELL,Y )

    # dataset_all=TensorDataset(M,G,EM,WELL,Y)
    dataset_all=TensorDataset(M,G,EM,WELL,Y)
    return dataset_all
def test_func(path1,path2,path4,path5,path3,number):

    M_ori = np.load(path2)
    G_ori = np.load(path1)
    EM_ori = np.load(path4)
    WELL_ori = np.load(path5)
    Y_ori = np.load(path3)
    M_ori = M_ori[:,:,:,:]
    G_ori = G_ori[:,:,:,:]
    EM_ori = EM_ori[:,:,:,:]
    WELL_ori = WELL_ori[:,:,:]
    Y_ori = Y_ori[:,:,:,:]

    train_M=M_ori.reshape([number,-1])
    train_M = train_M.reshape([number,1,10,10])
    train_G=G_ori.reshape([number,-1])
    train_G = train_G.reshape([number,1,20,20])
    train_EM=EM_ori.reshape([number,-1])
    train_EM = train_EM.reshape([number,2,10,5,8])


    train_WELL=WELL_ori.reshape([number,-1])
    train_WELL = train_WELL.reshape([number,2,20])

    train_Y=Y_ori.reshape([number,-1])
    train_Y = train_Y.reshape([number,8000])


    train_M=train_M.astype(np.float32)
    train_G=train_G.astype(np.float32)
    train_EM=train_EM.astype(np.float32)
    train_WELL=train_WELL.astype(np.float32)
    train_Y=train_Y.astype(np.float32)
    M = torch.from_numpy(train_M)
    G = torch.from_numpy(train_G)
    EM = torch.from_numpy(train_EM)
    WELL = torch.from_numpy(train_WELL)
    Y=torch.from_numpy(train_Y)
    
    # dataset_all=TensorDataset(M,G,EM,WELL,Y )

    # dataset_all=TensorDataset(M,G,EM,WELL,Y)
    dataset_all=TensorDataset(M,G,EM,WELL,Y)
    return dataset_all

def loaddata_func1(path1,path3,number):

    # M_ori = np.load(path2)
    G_ori = np.load(path1)
    Y_ori = np.load(path3)
    G_ori = G_ori[:1000,:,:,:]
    Y_ori = Y_ori[:1000,:,:,:]

    # train_M=M_ori.reshape([number,-1])
    # train_M = train_M.reshape([number,1,10,10])
    
    train_G=G_ori.reshape([number,-1])
    train_G = train_G.reshape([number,2,10,10])
    train_Y=Y_ori.reshape([number,-1])
    train_Y = train_Y.reshape([number,20,20,20])
    # train_M=train_M.astype(np.float32)
    train_G=train_G.astype(np.float32)
    train_Y=train_Y.astype(np.float32)
    # M = torch.from_numpy(train_M)
    G = torch.from_numpy(train_G)
    Y=torch.from_numpy(train_Y)
    
    dataset_all=TensorDataset(G,Y )
    # print(dataset_all.shape)
    return dataset_all

def shuffle_func(dataset,shuffle_ratio = 0.8):
    train_set_size = int(len(dataset) * shuffle_ratio)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])
    return train_set,valid_set