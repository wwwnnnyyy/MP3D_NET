from torch import optim
import torch.nn as nn
import torch
# import matplotlib.pyplot as plt
def valid(model,device,loader):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
#     criterion = nn.MSELoss()
    correct = 0
    total = len(loader.dataset)
    for x,x1,x2,x3,y in loader:
        x,y = x.to(device),y.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        with torch.no_grad():
            pred = model(x,x1,x2,x3) 
            loss = criterion(pred,y)
            print('test loss:',loss)

    return loss