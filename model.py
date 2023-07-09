import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv0(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down0(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            DoubleConv0(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up0(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=1, padding=1)

        self.conv = DoubleConv0(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv0, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels , kernel_size=3)
        
        # self.fc1 = nn.Linear(19200, 16000)
    def forward(self, x):
        # input_size = x.size(0)
        x = self.conv(x)

        return x



# %%
""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

class UNet0(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 20, bilinear=False):
        super(UNet0, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv0(n_channels, 64)
        self.down1 = Down0(64, 128)
        self.down2 = Down0(128, 256)
        self.down3 = Down0(256, 512)
        self.down4 = Down0(512, 512)
        self.up1 = Up0(1024, 256, bilinear)
        self.up2 = Up0(512, 128, bilinear)
        self.up3 = Up0(256, 64, bilinear)
        self.up4 = Up0(128, 64, bilinear)
        self.outc = OutConv0(64, 20)
#         self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = logits.view(logits.size(0),-1)
#         out = self.relu(logits)
#         out = self.sigmoid(logits)
        # result = torch.as_tensor((out - 0.5) > 0, dtype=torch.int64) 
        return out
    
class DoubleConv1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv1(x)


class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            DoubleConv1(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=2, padding=1)

        self.conv = DoubleConv1(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels , kernel_size=5)
        
        # self.fc1 = nn.Linear(19200, 16000)
    def forward(self, x):
        # input_size = x.size(0)
        x = self.conv(x)

        return x



# %%
""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

class UNet1(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 20, bilinear=False):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1(n_channels, 64)
        self.down1 = Down1(64, 128)
        self.down2 = Down1(128, 256)
        self.down3 = Down1(256, 512)
        self.down4 = Down1(512, 512)
        self.up1 = Up1(1024, 256, bilinear)
        self.up2 = Up1(512, 128, bilinear)
        self.up3 = Up1(256, 64, bilinear)
        self.up4 = Up1(128, 64, bilinear)
        self.outc = OutConv1(64, 20)
#         self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = logits.view(logits.size(0),-1)
#         out = self.relu(logits)
#         out = self.sigmoid(logits)
        # result = torch.as_tensor((out - 0.5) > 0, dtype=torch.int64) 
        return out

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2
            
        self.double_conv2 = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=(3,3,3), padding=2),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv2(x)


class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            DoubleConv2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up1 = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2, padding=1)

        self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        # input is CHW
        diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffX = torch.tensor([x2.size()[4] - x1.size()[4]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # x1(256,512,5,4,4) x2(5,4,4)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels , kernel_size=(5,5,5))
        
        # self.fc1 = nn.Linear(19200, 16000)
    def forward(self, x):
        # input_size = x.size(0)
        x = self.conv(x)

        return x

class UNet2(nn.Module):   ### AMT channel
    def __init__(self, n_channels = 2, n_classes = 20, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv2(n_channels, 64)
        self.down1 = Down2(64, 128)
        self.down2 = Down2(128, 256)
        self.down3 = Down2(256, 512)
#         self.down4 = Down2(512, 512)
        self.up1 = Up2(512, 256, bilinear=0)
        self.up2 = Up2(256, 128, bilinear=0)
        self.up3 = Up2(128, 64, bilinear=0)
        self.outc = OutConv2(64, 20)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.inc(x)       # x(2,5,8,10)  (64,7,10,12)
        x2 = self.down1(x1)    #(128,5,7,8)
        x3 = self.down2(x2)    #(256,4,5,6)
        x4 = self.down3(x3)    #(512,4,4,5)
#         x5 = self.down4(x4)    #(1028,4,4,4)
        x = self.up1(x4, x3)   #(512,6,7,8)
        x = self.up2(x, x2)    #(128,7,9,10)
        x = self.up3(x, x1)    #(64,9,12,14)
#         x = self.up4(x, x1)    #(64,14,9,12)
        logits = self.outc(x)  #(256,20,5,8,10)
        out = logits.view(logits.size(0),-1)   #(256,8000)
        # out = self.sigmoid(logits)
        # result = torch.as_tensor((out - 0.5) > 0, dtype=torch.int64) 
        return out


class DoubleConv3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            DoubleConv3(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up3 = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=3, stride=2, padding=1)

        self.conv = DoubleConv3(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up3(x1) #(256,11) #(256,8)

        diff = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels , kernel_size=5)
        
        # self.fc1 = nn.Linear(19200, 16000)
    def forward(self, x):
        # input_size = x.size(0)
        x = self.conv(x)

        return x

class UNet3(nn.Module):
    def __init__(self, n_channels = 2, n_classes = 400, bilinear=False):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3(n_channels, 64)
        self.down1 = Down3(64, 128)
        self.down2 = Down3(128, 256)
        self.down3 = Down3(256, 512)
        self.down4 = Down3(512, 512)
        self.up1 = Up3(1024, 256, bilinear=0)
        self.up2 = Up3(512, 128, bilinear=0)
        self.up3 = Up3(256, 64, bilinear=0)
        self.up4 = Up3(128, 64, bilinear=0)
        self.outc = OutConv3(64, n_classes)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = logits.view(logits.size(0),-1)   #(256,8000)
        # out = self.sigmoid(logits)
        # result = torch.as_tensor((out - 0.5) > 0, dtype=torch.int64) 
        return out       

    
class Unet1(nn.Module):
    def __init__(self):
        super(Unet1, self).__init__()
        self.unet0 = UNet0()
        self.unet1 = UNet1()
        self.unet2 = UNet2()
        self.unet3 = UNet3()
        self.sigmoid = nn.Sigmoid()

         
    # def forward(self, x0, x1):
    def forward(self, x0,x1,x2,x3):
        y0 = self.unet0(x0)
        y1 = self.unet1(x1)
        y2 = self.unet2(x2)
        y3 = self.unet3(x3)

        out0 = self.sigmoid(y0)     #mag(256,8000)
        out1 = self.sigmoid(y1)     #gra(256,8000)
        out2 = self.sigmoid(y2)     #amt(256,8000)
        out3 = self.sigmoid(y3)     #well(256,8000)
        
        result = 1/4*(out0+out1+out2+out3) #########(32,8000)
#         out = result.view([32,20,20,20])   #(256,20,20,20)
        # result = torch.as_tensor((out - 0.5) > 0, dtype=torch.int64)
        return result
       


if __name__ == '__main__':
    net1 = Unet1()
#     print(net1)