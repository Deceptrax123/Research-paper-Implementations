import torch
from torch import nn
import torchvision

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d,Linear,Flatten

from prettytable import PrettyTable
from torchsummary import summary

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()

        #project noise of latent space to a 4d stack
        self.linear=Linear(in_features=100,out_features=16384)

        self.conv1 = ConvTranspose2d(in_channels=1024, out_channels=512,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn1=BatchNorm2d(512)
        self.relu1 = ReLU()

        self.conv2 = ConvTranspose2d(in_channels=512, out_channels=256,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn2 = BatchNorm2d(num_features=256)
        self.relu2 = ReLU()

        self.conv3 = ConvTranspose2d(in_channels=256, out_channels=128,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn3 = BatchNorm2d(num_features=128)
        self.relu3 = ReLU()

        self.conv4 = ConvTranspose2d(in_channels=128, out_channels=64,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn4=BatchNorm2d(num_features=64)
        self.relu4=ReLU()

        self.conv5=ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(4,4),stride=2,padding=1)
        self.bn5=BatchNorm2d(num_features=32)
        self.relu5=ReLU()

        self.conv6=ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=(4,4),stride=2,padding=1)
        self.bn6=BatchNorm2d(3)
        self.relu6=ReLU()

        #bottle neck convolutions
        self.bot1=ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=1,padding=1)
        self.bn7=BatchNorm2d(3)
        self.relu7=ReLU()

        self.bot2=ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=1,padding=1)
        self.bn8=BatchNorm2d(3)
        self.relu8=ReLU()

        self.bot3=ConvTranspose2d(in_channels=3,out_channels=3,stride=1,padding=1,kernel_size=(3,3))
        self.bn9=BatchNorm2d(3)
        self.relu9=ReLU()

        self.bot4=ConvTranspose2d(in_channels=3,out_channels=3,stride=1,padding=1,kernel_size=(3,3))
        self.bn10=BatchNorm2d(3)
        self.relu10=ReLU()

        self.bot5=ConvTranspose2d(in_channels=3,out_channels=3,stride=1,padding=1,kernel_size=(3,3))
        self.bn11=BatchNorm2d(3)
        self.relu11=ReLU()

        self.bot6=ConvTranspose2d(in_channels=3,out_channels=3,stride=1,padding=1,kernel_size=(3,3))
        self.tanh = Tanh()

    def forward(self, x):
        x=self.linear(x)
        x=x.view(x.size(0),1024,4,4)

        x = self.conv1(x)
        x = self.bn1(x)
        x= self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x=self.bn4(x)
        x=self.relu4(x)

        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu5(x)

        x=self.conv6(x)
        x=self.bn6(x)
        x=self.relu6(x)

        x=self.bot1(x)
        x=self.bn7(x)
        x=self.relu7(x)

        x=self.bot2(x)
        x=self.bn8(x)
        x=self.relu8(x)

        x=self.bot3(x)
        x=self.bn9(x)
        x=self.relu9(x)


        x=self.bot4(x)
        x=self.tanh(x)

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


# model = Generator()
# summary(model,input_size=(100,),batch_size=8,device='cpu')