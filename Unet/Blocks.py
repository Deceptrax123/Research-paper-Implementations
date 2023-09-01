import torch 
from torch.nn import Conv2d,ConvTranspose2d,ReLU,BatchNorm2d,MaxPool2d,MaxUnpool2d,AdaptiveAvgPool2d,Upsample,Dropout2d,LeakyReLU
from torch.nn import Module 
from torchsummary import summary
from torch import nn

class Unet_encoding_block(Module):
    def __init__(self,features):
        super(Unet_encoding_block,self).__init__()

        self.conv1=Conv2d(in_channels=features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(2*features)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=2*features,out_channels=2*features,stride=1,kernel_size=(3,3),padding=1)
        self.bn2=BatchNorm2d(2*features)
        self.relu2=ReLU()

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,(nn.Conv2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        return x

class Unet_decoding_block(Module):
    def __init__(self,features):
        super(Unet_decoding_block,self).__init__()

        self.dconv1=ConvTranspose2d(in_channels=features,out_channels=features//2,stride=1,padding=1,kernel_size=(3,3))
        self.bn1=BatchNorm2d(features//2)
        self.relu1=ReLU()

        self.dconv2=ConvTranspose2d(in_channels=features//2,out_channels=features//2,kernel_size=(3,3),padding=1,stride=1)
        self.bn2=BatchNorm2d(features//2)
        self.relu2=ReLU()

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(nn.ConvTranspose2d,nn.BatchNorm2d)):
            if module.bias.data is not None :
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(module.weight.data,mode='fan_in',nonlinearity='relu')

    def forward(self,x):
        x=self.dconv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        
        x=self.dconv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        return x
