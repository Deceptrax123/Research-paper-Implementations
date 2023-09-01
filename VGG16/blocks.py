from torch.nn import Conv2d,MaxPool2d,Linear,BatchNorm2d,ReLU,Softmax
from torch.nn import Module
from torchsummary import summary
from torch.nn.init import kaiming_normal_,xavier_normal_

class Double_Convolution_Layer(Module):
    def __init__(self,filters,input=3):
        super(Double_Convolution_Layer,self).__init__()

        self.conv1=Conv2d(in_channels=input,out_channels=filters,kernel_size=(3,3),padding=1,stride=1)
        self.bn1=BatchNorm2d(filters)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=filters,out_channels=filters,kernel_size=(3,3),padding=1,stride=1)
        self.bn2=BatchNorm2d(filters)
        self.relu2=ReLU()

        self.pool=MaxPool2d(kernel_size=(2,2),stride=2)

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,(Conv2d,BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu')

    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.pool(x)

        return x

class Triple_Convolution_Layer(Module):
    def __init__(self,filters,input):
        super(Triple_Convolution_Layer,self).__init__()

        self.conv1=Conv2d(in_channels=input,out_channels=filters,kernel_size=(3,3),stride=1,padding=1)
        self.bn1=BatchNorm2d(filters)
        self.relu1=ReLU()

        self.conv2=Conv2d(in_channels=filters,out_channels=filters,kernel_size=(3,3),stride=1,padding=1)
        self.bn2=BatchNorm2d(filters)
        self.relu2=ReLU()

        self.conv3=Conv2d(in_channels=filters,out_channels=filters,stride=1,padding=1,kernel_size=(3,3))
        self.bn3=BatchNorm2d(filters)
        self.relu3=ReLU()

        self.pool=MaxPool2d(kernel_size=(2,2),stride=2)

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,(Conv2d,BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu')
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)

        x=self.pool(x)
        
        return x

class Fully_Connected(Module):
    def __init__(self,in_f,out_f):
        super(Fully_Connected,self).__init__()

        self.ln1=Linear(in_features=in_f,out_features=out_f)
        self.bn1=BatchNorm2d(out_f)
        self.relu1=ReLU()

        self.apply(self._init_weights)
    def _init_weights(self,module):
         if isinstance(module,(Linear,BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu')
    def forward(self,x):
        x=self.ln1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        return x

class Classifer(Module):
    def __init__(self,in_f,out_f):
        super(Classifer,self).__init__()

        self.ln=Linear(in_features=in_f,out_features=out_f)
        self.softmax=Softmax()
    
    def _init_weights(self,module):
        if isinstance(module,(Linear)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                xavier_normal_(module.weight.data)
                
    def forward(self,x):
        x=self.ln(x)
        x=self.softmax(x)

        return x
