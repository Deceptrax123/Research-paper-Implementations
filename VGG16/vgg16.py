from torch.nn import Module 
from blocks import Double_Convolution_Layer,Triple_Convolution_Layer,Fully_Connected,Classifer
from torchsummary import summary

#To use Dense Layers : Input size must be (...,3,224,224) where ... is batch size

class VGG16(Module):
    def __init__(self):
        super(VGG16,self).__init__()

        self.dc1=Double_Convolution_Layer(filters=64)
        self.dc2=Double_Convolution_Layer(input=64,filters=128)

        self.tc1=Triple_Convolution_Layer(input=128,filters=256)
        self.tc2=Triple_Convolution_Layer(input=256,filters=512)
        self.tc3=Triple_Convolution_Layer(input=512,filters=512)

        self.dense1=Fully_Connected(in_f=7*7*512,out_f=4096)
        self.dense2=Fully_Connected(in_f=4096,out_f=4096)

        self.classifier=Classifer(in_f=4096,out_f=1000)
    
    def forward(self,x):
        x=self.dc1(x)
        x=self.dc2(x)

        x=self.tc1(x)
        x=self.tc2(x)
        x=self.tc3(x)
        
        x=x.view(x.size(0),x.size(1)*x.size(2)*x.size(3))
        x=self.dense1(x)
        x=self.dense2(x)
        
        x=self.classifier(x)

        return x