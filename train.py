from torch import optim
import torch.nn as nn
import torch
import os
from valid import*
# import matplotlib.pyplot as plt
# from earlyval import *
from tensorboardX import SummaryWriter
import datetime
nowtime = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
print(nowtime + '\n')
pwd = os.getcwd()+'/' + 'log/' + nowtime
isExists = os.path.exists(pwd)
if not isExists:
    os.makedirs(pwd)
writer = SummaryWriter(pwd)


def train_net(net, device, dataloader1, dataloader3,dataloader5,epochs, batch_size, lr):
    # 加载训练集
   
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#     optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
#     criterion = nn.MSELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
#         for image, label in train_loader:
        for (i, (image0,image1,image2,image3, label)) in enumerate(dataloader1):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image0 = image0.to(device=device, dtype=torch.float32)
            image1 = image1.to(device=device, dtype=torch.float32)
            image2 = image2.to(device=device, dtype=torch.float32)
            image3 = image3.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred0 = net(image0,image1,image2,image3)
            # pred1 = net(image1)

            result0 = torch.as_tensor((pred0 - 0.5) > 0, dtype=torch.int64) 
            # result1 = torch.as_tensor((pred1 - 0.5) > 0, dtype=torch.int64) 
#             result = pred0.detach().numpy()
#             label_ = label.numpy()

            loss0 = criterion(pred0, label)
#             loss1 = criterion(pred1, label1)
            loss = loss0
            niter = epoch*len(dataloader1)+i
            if niter % 2 == 0:
                val_acc = valid(net,device,dataloader3)
                # print('val acc :', val_acc)
                
                writer.add_scalar('Val loss',val_acc,niter)
                writer.add_scalar('Train loss',loss,niter)
                writer.add_graph(net,(image0,image1,image2,image3,))
                writer.flush()
                test_acc = valid(net,device,dataloader5)
                writer.add_scalar('Test loss',test_acc,niter)
                writer.flush()
            print('Loss/train', loss.item())

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model90.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        print('epoch: {} '.format(epoch))
    # print('val acc :', val_acc)
    
