import torch 
from torch import nn 

from torch.nn import Module 
from torch.nn import Conv2d,BatchNorm2d,Flatten,Sigmoid,LeakyReLU,ConvTranspose2d,Linear
from prettytable import PrettyTable

from torchsummary import summary

class Discriminator(Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=1)
        self.lr1=LeakyReLU(negative_slope=0.2)

        self.conv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(num_features=16)
        self.lr2=LeakyReLU(negative_slope=0.2)

        self.conv3=Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.bn3=BatchNorm2d(32)
        self.lr3=LeakyReLU(negative_slope=0.2)

        self.conv4=Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1)
        self.bn4=BatchNorm2d(64)
        self.lr4=LeakyReLU(negative_slope=0.2)

        self.flatten=Flatten(start_dim=1)
        self.linear=Linear(4096,1)

        self.sigmoid=Sigmoid()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.lr1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.lr2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.lr3(x)

        x=self.conv4(x)
        x=self.bn4(x)
        x=self.lr4(x)

        x=self.flatten(x)
        x=self.linear(x)
        x=self.sigmoid(x)

        return x

# def count_parameters(model):
#     table = PrettyTable(['Modules', 'Parameters'])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f'Total Trainable Params: {total_params}')
#     return total_params


# model = Discriminator()
# # # count_parameters(model)

# summary(model,input_size=(3,128,128),batch_size=8,device='cpu')